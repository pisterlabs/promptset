import openai
import pymongo

from constants import OPENAI_API_KEY


def embed_text(text: str, model: str = "text-embedding-ada-002") -> list[float]:
    """Embeds `text` using `model`."""
    text = text.replace("\n", " ")
    return openai.Embedding.create(
        input=[text],
        model=model,
        api_key=OPENAI_API_KEY,
    )[
        "data"
    ][0]["embedding"]


def add_document(
    document: str,
    collection: pymongo.collection.Collection,
    embeddings_field: str,
    embeddings_model: str = "text-embedding-ada-002",
) -> None:
    """Adds `document` to the vector search database."""
    embeddings = embed_text(document, embeddings_model)
    collection.insert_one(
        {
            "text": document,
            embeddings_field: embeddings,
        }
    )


def vector_search(
    query: str,
    collection: pymongo.collection.Collection,
    index: str,
    embeddings_field: str,
    candidates: int = 100,
    limit: int = 4,
    embeddings_model: str = "text-embedding-ada-002",
) -> list[str]:
    """Finds the most similar documents to `query` in `collection` using
    vector search.
    """
    query_vector = embed_text(query, embeddings_model)
    results = collection.aggregate(
        [
            {
                "$vectorSearch": {
                    "queryVector": query_vector,
                    "path": embeddings_field,
                    "numCandidates": candidates,
                    "limit": limit,
                    "index": index,
                }
            }
        ]
    )
    return [result["text"] for result in results]
