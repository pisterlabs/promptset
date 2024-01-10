import lancedb
import torch
import pyarrow as pa
import pandas as pd
from pathlib import Path
import tqdm
import numpy as np
import cohere
from os import getenv
from sentence_transformers import SentenceTransformer
# from gradio_app.constants import (DB_TABLE_NAME, VECTOR_COLUMN_NAME, TEXT_COLUMN_NAME, FILES_DUMP_FOLDER)

cohere_embedding_dimensions = {
    "embed-english-v3.0": 1024,
    "embed-multilingual-v3.0": 1024,
    "embed-english-light-v3.0": 384,
    "embed-multilingual-light-v3.0": 384,
    "embed-english-v2.0": 4096,
    "embed-english-light-v2.0": 1024,
    "embed-multilingual-v2.0": 768,
}

EMB_MODEL_NAME = "paraphrase-albert-small-v2"
EMB_MODEL_NAME = "thenlper/gte-large"
EMB_MODEL_NAME = "all-MiniLM-L6-v2"
# EMB_MODEL_NAME = "embed-english-v3.0"
# EMB_MODEL_NAME = "all-mpnet-base-v2"

if EMB_MODEL_NAME in ["paraphrase-albert-small-v2", "all-MiniLM-L6-v2", "all-mpnet-base-v2"]:
    mode = 'ST'
elif EMB_MODEL_NAME in list(cohere_embedding_dimensions.keys()):
    mode = 'COHERE'
else:
    mode = None

DB_TABLE_NAME = "split_files_db"
VECTOR_COLUMN_NAME = "vctr"
TEXT_COLUMN_NAME = "txt"
FILES_DUMP_FOLDER = "split_files_dump"


INPUT_DIR = FILES_DUMP_FOLDER
db = lancedb.connect("gradio_app/.lancedb")
batch_size = 32

if mode == 'ST':
    model = SentenceTransformer(EMB_MODEL_NAME)
    model.eval()
    embedding_size = model.get_sentence_embedding_dimension()
elif mode == 'COHERE':
    co = cohere.Client(getenv('COHERE_API_KEY'))
    embedding_size = cohere_embedding_dimensions[EMB_MODEL_NAME]
else:
    embedding_size = None

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


schema = pa.schema(
  [
      pa.field(VECTOR_COLUMN_NAME, pa.list_(pa.float32(), embedding_size)),
      pa.field(TEXT_COLUMN_NAME, pa.string())
  ])
tbl = db.create_table(DB_TABLE_NAME, schema=schema, mode="overwrite")

input_dir = Path(INPUT_DIR)
files = list(input_dir.rglob("*"))

sentences = []
for file in files:
    with open(file) as f:
        sentences.append(f.read())

for i in tqdm.tqdm(range(0, int(np.ceil(len(sentences) / batch_size)))):
    try:
        batch = [sent for sent in sentences[i * batch_size:(i + 1) * batch_size] if len(sent) > 0]

        if mode == 'ST':
            encoded = model.encode(batch, normalize_embeddings=True, device=device)
        elif mode == 'COHERE':
            encoded = np.array(co.embed(batch, input_type="search_document", model="embed-english-v3.0").embeddings)
        else:
            encoded = None
        encoded_lst = [list(vec) for vec in encoded]

        df = pd.DataFrame({
            VECTOR_COLUMN_NAME: encoded_lst,
            TEXT_COLUMN_NAME: batch
        })

        tbl.add(df)
    except Exception as e:
        print(f"batch {i} raised an exception: {str(e)}")

'''
create ivf-pd index https://lancedb.github.io/lancedb/ann_indexes/
with the size of the transformer docs, index is not really needed
but we'll do it for demonstrational purposes
'''
tbl.create_index(num_partitions=256, num_sub_vectors=96, vector_column_name=VECTOR_COLUMN_NAME)
