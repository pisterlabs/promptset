import redis
import openai
import redis
from redis.commands.search.query import Query
from redis.commands.search.document import Document
import numpy as np

INDEX_NAME = "embeddings-wmd-index"

# Run a search query and return the results
from typing import List
def search(
        redis_client: redis.Redis,
        user_query: str,
        index_name: str = INDEX_NAME,
        vector_field: str = "embedding",
        return_fields: list = ["topic", "overview", "symptoms", "url", "vector_score"],
        hybrid_fields = "*",
        k: int = 20,
        print_results: bool = False,
) -> List[Document]:
    """
    Search Redis for a given query and return the results.
    :param redis_client: Redis client
    :param user_query: Query string
    :param index_name: Name of the index to search in
    :param vector_field: Name of the vector field
    :param return_fields: List of fields to return
    :param hybrid_fields: List of fields to use for hybrid search
    :param k: Number of results to return
    :param print_results: Whether to print the results
    :return: List of results
    """
    # Creates embedding vector from user query
    embedded_query = openai.Embedding.create(input=user_query,
                                             model="text-embedding-ada-002",
                                             )["data"][0]['embedding']
    
    # Prepare the query
    base_query = f'{hybrid_fields}=>[KNN {k} @{vector_field} $vector AS vector_score]'

    query = (
        Query(base_query)
        .return_fields(*return_fields)
        .sort_by("vector_score")
        .paging(0, k)
        .dialect(2)
    )

    params_dict = {
        "vector": np.array(embedded_query).astype(dtype=np.float32).tobytes()
    }

    # perforrm vector search
    results = redis_client.ft(index_name).search(query, params_dict)

    # Print the results
    if print_results:
        for i, result in enumerate(results.docs):
            print(f"Rank: {i}")
            print(f"Topic: {result.topic}")
            print(f"Overview: {result.overview}")
            print(f"Symptoms: {result.symptoms}")
            print(f"URL: {result.url}")
            score = 1 - float(result.vector_score)
            print(f"Score: {round(score, 3)})")
            print()

    return results.docs