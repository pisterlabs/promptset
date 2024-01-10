from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from openai import OpenAI
import os
import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)

MODEL = os.environ.get('MODEL', 'text-embedding-ada-002')
BATCH_SIZE = 512

client = OpenAI()

def embed(labels):
    for batch in range(0, len(labels), BATCH_SIZE):
        for embedding in client.embeddings.create(
            input = labels[batch:batch+BATCH_SIZE], 
            model=MODEL).data:
            yield embedding.embedding 

eventtypes = pd.read_parquet('mimicseq/eventtypes.parquet')
eventtypes.sort_index(inplace=True)

embedding = np.stack(list(embed(eventtypes.label.to_list())))
                             
clusters = np.stack([
    KMeans(n_clusters=10, n_init=5, init='random').fit_predict(embedding),
    KMeans(n_clusters=100, n_init=5, init='random').fit_predict(embedding),
    KMeans(n_clusters=1000, n_init=5, init='random').fit_predict(embedding),
    KMeans(n_clusters=10000, n_init=5, init='random').fit_predict(embedding)
])

eventtypes[['c10', 'c100', 'c1000', 'c10000']] = clusters.T
eventtypes.to_parquet('mimicseq/eventtypes_.parquet')