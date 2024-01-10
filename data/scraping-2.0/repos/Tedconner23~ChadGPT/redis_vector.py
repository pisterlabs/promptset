import redis
import openai
import ast
from math import isnan
import numpy as np
import pandas as pd
from redis import Redis as r
from redis.commands.search.query import Query

from config import (
    CONVERSATION_DB,
    PERSONA_DB,
    MEMORY_DB,
    CONTENT_DB,
    REDIS_HOST,
    REDIS_PORT,
    VECTOR_FIELD_NAME,
    EMBEDDINGS_MODEL,
    INDEX_NAME,
)

class RedisDatabase:
    """
    Class representing the Redis databases.
    """

    def __init__(self, db):
        """
        Initializes the Redis databases.
        """
        self.redis_conn = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=db)

    def save_data(self, key, data):
        """
        Saves data to the Redis database.
        """
        self.redis_conn.set(key, data)

    def get_data(self, key):
        """
        Retrieves data from the Redis database.
        """
        data = self.redis_conn.get(key)
        return data

    def delete_data(self, key):
        """
        Deletes data from the Redis database.
        """
        self.redis_conn.delete(key)

    def update_data(self, key, new_data):
        """
        Updates data in the Redis database.
        """
        self.redis_conn.set(key, new_data)

    def save_list(self, key, data_list):
        """
        Saves a list to the Redis database.
        """
        self.redis_conn.rpush(key, *data_list)

    def get_list(self, key, start=0, end=-1):
        """
        Retrieves a list from the Redis database.
        """
        data_list = self.redis_conn.lrange(key, start, end)
        return data_list

    def search_keys(self, pattern):
        """
        Searches for keys in the Redis database using a pattern.
        """
        matching_keys = self.redis_conn.keys(pattern)
        return matching_keys

conversationDB = RedisDatabase(CONVERSATION_DB)
personaDB = RedisDatabase(PERSONA_DB)
memoryDB = RedisDatabase(MEMORY_DB)
contentDB = RedisDatabase(CONTENT_DB)

class EmbeddingModel:
    """
    Class representing the embedding model.
    """

    def __init__(self):
        """
        Initializes the embedding model.
        """
        pass

    def embed_text(self, text):
        """
        Generates an embedding for the given text.
        """
        embedding = openai.Embedding.create(input=text, model=EMBEDDINGS_MODEL)["data"][0]["embedding"]
        return embedding

def get_redis_connection():
    redis_client = r(
        host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=False
    )
    return redis_client

def query_redis(redis_conn, query, index_name, top_k=5):
    embedded_query = np.array(
        openai.Embedding.create(input=query, model=EMBEDDINGS_MODEL)["data"][0]["embedding"],
        dtype=np.float32,
    ).tobytes()

    q = (
        Query(f"*=>[KNN {top_k} @{VECTOR_FIELD_NAME} $vec_param AS vector_score]")
        .sort_by("vector_score")
        .paging(0, top_k)
        .return_fields("vector_score", "url", "title", "content", "text_chunk_index")
        .dialect(2)
    )
    params_dict = {"vec_param": embedded_query}

    results = redis_conn.ft(index_name).search(q, query_params=params_dict)

    return results

def get_redis_results(redis_conn, query, index_name):
    query_result = query_redis(redis_conn, query, index_name)

    query_result_list = []
    for i, result in enumerate(query_result.docs):
        result_order = i
        url = result.url
        title = result.title
        text = result.content
        score = result.vector_score
        query_result_list.append((result_order, url, title, text, score))

    result_df = pd.DataFrame(query_result_list)
    result_df.columns = ["id", "url", "title", "result", "certainty"]
    return result_df
