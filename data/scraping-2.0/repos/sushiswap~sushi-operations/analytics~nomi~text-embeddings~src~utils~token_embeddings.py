
import openai
import pandas as pd
import numpy as np
import os
from utils.config import OPEN_API_KEY

EMBEDDING_MODEL = "text-embedding-ada-002"

openai.api_key = OPEN_API_KEY


def get_embedding(text, model=EMBEDDING_MODEL):
    return openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']


def compute_doc_embeddings(df: pd.DataFrame) -> dict[(str, str), np.array]:
    return {
        idx: get_embedding(r.content) for idx, r in df.iterrows()
    }


def vector_similarity(x: list[float], y: list[float]) -> float:
    """
    Returns the similarity between two vectors.

    Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    """
    return np.dot(np.array(x), np.array(y))


def order_by_similarity(query: str, contexts: dict[(str, str), np.array]) -> list[(float, (str, str))]:
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections.

    Return the list of document sections, sorted by relevance in descending order.
    """
    query_embedding = get_embedding(query)

    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)

    return document_similarities
