from typing import List, Tuple, Optional
from scipy import spatial
import openai

from backend.embedding_types import EmbeddingRecord
from backend.oai.setup import EMBEDDING_MODEL, setup_openai

setup_openai()


# Function to compute the relatedness using cosine similarity
def relatedness_fn(x: List[float], y: List[float]) -> float:
    try:
        return 1 - spatial.distance.cosine(x, y)
    except Exception as e:
        print(f"Error in relatedness_fn: {e}")
        return float("-inf")  # Return negative infinity for error cases


# Function to fetch query embedding
def fetch_query_embedding(query: str) -> Optional[List[float]]:
    try:
        query_embedding_response = openai.Embedding.create(
            model=EMBEDDING_MODEL, input=query
        )
        return query_embedding_response["data"][0]["embedding"]
    except Exception as e:
        print(f"Error in fetch_query_embedding: {e}")
        return None


# The main function for context retrieval
def retrieve_context(
    query: str, embedding_records: List[EmbeddingRecord], k: int = 5
) -> Tuple[Optional[List[float]], List[EmbeddingRecord]]:
    query_embedding = fetch_query_embedding(query)

    if query_embedding is None:
        return None, []

    # Calculate relatedness for each record
    records_and_relatedness = [
        (record, relatedness_fn(query_embedding, record.embedding))
        for record in embedding_records
    ]

    # Sort records by relatedness
    records_and_relatedness.sort(key=lambda x: x[1], reverse=True)

    # Extract the k most related records
    closest_k_records = [record for record, _ in records_and_relatedness[:k]]

    return query_embedding, closest_k_records
