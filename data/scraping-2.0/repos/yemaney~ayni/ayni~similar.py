"""This module is concerned with code related to finding the most similar context for given queries
"""

from typing import List

import numpy as np
import openai
from scipy import spatial


def get_embedding(text: str, engine: str) -> List[float]:
    # replace newlines, which can negatively affect performance.
    text = text.replace("\n", " ")

    return openai.Embedding.create(input=[text], engine=engine)["data"][0]["embedding"]


def distances_from_embeddings(
    query_embedding: List[float],
    embeddings: List[List[float]],
    distance_metric="cosine",
) -> List[List]:
    """Return the distances between a query embedding and a list of embeddings."""
    distance_metrics = {
        "cosine": spatial.distance.cosine,
        "L1": spatial.distance.cityblock,
        "L2": spatial.distance.euclidean,
        "Linf": spatial.distance.chebyshev,
    }
    distances = [distance_metrics[distance_metric](query_embedding, embedding) for embedding in embeddings]
    return distances


def indices_of_nearest_neighbors_from_distances(distances) -> np.ndarray:
    """Return a list of indices of nearest neighbors from a list of distances."""
    return np.argsort(distances)


def get_most_similar(query: str, model: str, embedding_cache: dict) -> str:
    """get_most_similar finds the most similar function in the embedding_cache to a given query.

    Parameters
    ----------
    query : str
        A string representing a question a user has for the code
    model : str
        The name of the openai embedding model to use for generating embeddings
    embedding_cache : dict
        A dictionary containing the cached embeddings

    Returns
    -------
    string
        the most similar function to the query
    """
    query_embedding = get_embedding(query, model)

    functions = [k for k, _ in embedding_cache.items()]
    embeddings = [v for _, v in embedding_cache.items()]

    distances = distances_from_embeddings(query_embedding, embeddings)
    indices_of_nearest_neighbors = indices_of_nearest_neighbors_from_distances(distances)

    return [functions[i] for i in indices_of_nearest_neighbors][0]
