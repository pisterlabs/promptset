
import numpy as np
import openai
from joblib import Memory

memory = Memory(".api_cache", verbose=0)
embed_model = "text-embedding-ada-002"


@memory.cache
def get_embeddings(texts):
    return [i['embedding'] for i in openai.Embedding.create(input=texts, engine=embed_model)['data']]


class TrafficCone:

    def __init__(self):
        self.embeddings = {}

    def insert(self, texts):
        for i, j, in zip(texts, get_embeddings(texts)):
            self.embeddings[i] = j

    def query(self, text):
        return sorted([
            (self.get_similarity(get_embeddings([text])[0], j), i)
            for i, j in self.embeddings.items()],
            reverse=True)

    def get_similarity(self, v1, v2):
        return np.dot(v1, v2)
