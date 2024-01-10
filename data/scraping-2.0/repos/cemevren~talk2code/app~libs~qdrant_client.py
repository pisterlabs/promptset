from langchain.vectorstores.qdrant import Qdrant
from langchain_core.embeddings import Embeddings
import qdrant_client
from qdrant_openapi_client.models import VectorParams, Distance


def get_qdrant_client(embedding_model: Embeddings, collection_name: str):
    client = qdrant_client.QdrantClient(
        url="http://localhost:6333",
    )

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        force_recreate=True,
    )

    qdrant = Qdrant(
        client=client, embeddings=embedding_model, collection_name=collection_name
    )
    return qdrant
