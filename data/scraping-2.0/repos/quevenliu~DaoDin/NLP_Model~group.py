import numpy as np
from k_means_constrained import KMeansConstrained
import time
import openai
from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("OPEN_API")
openai.api_key = api_key


def get_embedding(text, model):
    embedding = openai.Embedding.create(
        model=model,
        input=text,
    )
    return embedding['data'][0]['embedding']


def group(data, model):
    SIZE_MIN = 4
    SIZE_MAX = 5
    N_CLUSTERS = int(np.ceil(len(data) / SIZE_MAX))
    user_ids = []
    embeddings = []
    for user in data:
        user_ids.append(user['user_id'])
        embeddings.append(get_embedding(
            user['self_intro'] + '\n' + user['match_msg'], model))
    embeddings = np.array(embeddings)
    kmeans = KMeansConstrained(n_clusters=N_CLUSTERS,
                               size_min=SIZE_MIN,
                               size_max=SIZE_MAX).fit(embeddings)
    result = [[] for _ in range(N_CLUSTERS)]
    for i in range(len(user_ids)):
        result[kmeans.labels_[i]].append(user_ids[i])
    return result
