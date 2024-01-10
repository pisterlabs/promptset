from fastapi import APIRouter, HTTPException, Query
from app.services.openai import OpenAIService
from numpy import dot
from numpy.linalg import norm
from ..shared_resources import page_data_store

router = APIRouter()
_THRESHOLD = 0.80


@router.post("/")
async def find_similar_pages(
    query: str,
    ai_provider: str = Query("openai", description="AI provider for semantic search"),
):
    """
    Perform a semantic search on the document collection based on the given query.

    Parameters:
    - query (str): The query to search for.
    - ai_provider (str): The AI provider for semantic search. Default is "openai".

    Returns:
    - List[Document]: A list of documents that are semantically similar to the query.
    """
    if ai_provider == "openai":
        similar_pages = embed_then_compute(query, page_data_store)
    else:
        raise HTTPException(status_code=400, detail="Unsupported AI provider")
    return similar_pages


def encode(text):
    """
    Generate an embedding for the given text using OpenAI's API.

    Parameters:
    - text (str): The text to encode.

    Returns:
    - list: The embedding vector for the text.
    """
    openaiservice = OpenAIService()
    response = openaiservice.Embedding.create(
        input=text, engine="text-similarity-babbage-001"
    )
    return response["data"][0]["embedding"]


def embed_then_compute(query, page_data_store):
    # Convert query to an embedding
    query_embedding = encode(query)

    # Find semantically similar documents in the store
    similar_pages = []
    for _, doc in page_data_store.items():  # Use .items() to get both key and value
        page_embeddings = encode(doc["text"])
        similarity = _compute_similarity(query_embedding, page_embeddings)
        if similarity > _THRESHOLD:
            similar_pages.append(doc)

    return similar_pages


def _compute_similarity(emb1, emb2):
    # simple cosine similarity
    return dot(emb1, emb2) / (norm(emb1) * norm(emb2))
