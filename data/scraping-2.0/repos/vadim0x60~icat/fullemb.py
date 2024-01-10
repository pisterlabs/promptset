import openai
import numpy as np
import tiktoken as tk
import pandas as pd
from logwrap import logwrap
import tenacity as t

from equibatch import EquiarealBatchSampler

MODEL = "text-embedding-ada-002"
MAX_BATCH_SIZE = 2048
MAX_BATCH_FOORPRINT = 8191
EMBEDDING_DIM = 1536

@t.retry(wait=t.wait_random_exponential(), stop=t.stop_after_attempt(6))
def openai_embed(labels):
    data = openai.Embedding.create(
        model=MODEL,
        input=labels
    )['data']

    return np.array([datum['embedding'] for datum in data])

@logwrap
def all_labels(eventtypes, specials=['[PAD]', '[INTENSITY]']):
    label_count = len(eventtypes) + len(specials)
    labels = [None] * label_count

    special_ixs = {}
    specials = list(specials)

    # This is convoluted, but it gives us a guarantee
    # that label indices are the same in the dataset and in the embedding,
    # potentially preventing nasty indexing errors.
    for ix in range(label_count):
        try:
            labels[ix] = eventtypes.loc[ix]['label']
        except KeyError:
            labels[ix] = specials[0]
            special_ixs[specials[0]] = ix
            del specials[0]

    return labels, special_ixs

@logwrap
def fill_embedding(embedding, labels, callback):
    enc = tk.encoding_for_model(MODEL)
    
    labels = [enc.encode(label) for label in labels]
    # gimme that sweet, sweet numpy indexing
    labels = np.array(labels, dtype=np.object_)

    batch_sampler = EquiarealBatchSampler(
        np.where((embedding == 0).all(axis=1))[0],
        lambda ix: len(labels[ix]),
        MAX_BATCH_SIZE,
        MAX_BATCH_FOORPRINT
    )

    for batch in batch_sampler:
        logging.debug(f'Embedding labels {batch}')
        # sweet numpy indexing om nom nom
        embedding[batch] = openai_embed(labels[batch].tolist())
        callback(batch)

if __name__ == '__main__':
    import shutil
    import yaml

    import logging
    logging.basicConfig(level=logging.DEBUG)

    eventtypes = pd.read_parquet('mimicseq/eventtypes.parquet')
    labels, special_ixs = all_labels(eventtypes)

    with open('embeddings/special_ixs.yml', 'w') as f:
        yaml.dump(special_ixs, f)

    try:
        embedding = np.load('embeddings/full.temp.npy')
        if len(embedding) < len(labels):
            delta_shape = (len(labels) - len(embedding), EMBEDDING_DIM)
            delta = np.zeros(delta_shape, dtype=np.float32)
            embedding = np.vstack((embedding, delta))
    except FileNotFoundError:
        embedding = np.zeros((len(labels), EMBEDDING_DIM), dtype=np.float32)

    def flush(batch):
        np.save('embeddings/full.temp.npy', embedding)

    fill_embedding(embedding, labels, flush)
    shutil.move('embeddings/full.temp.npy', 'embeddings/full.npy')