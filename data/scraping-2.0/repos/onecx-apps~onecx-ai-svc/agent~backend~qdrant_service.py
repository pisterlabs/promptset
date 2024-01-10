import os
from langchain.vectorstores import Qdrant
from langchain.schema.embeddings import Embeddings
from qdrant_client import QdrantClient, models
from agent.utils.configuration import load_config
from loguru import logger

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
QDRANT_URL=os.getenv("QDRANT_URL")
QDRANT_PORT=os.getenv("QDRANT_PORT")
QDRANT_API_KEY=os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME=os.getenv("QDRANT_COLLECTION_NAME")


def get_db_connection(embedding_model: Embeddings) -> Qdrant:
    """get_db_connection initializes the connection to the Qdrant db.

    :return: Qdrant DB connection
    :rtype: Qdrant
    """
    embedding = embedding_model
    qdrant_client = QdrantClient(url=QDRANT_URL, port=QDRANT_PORT, api_key=QDRANT_API_KEY, prefer_grpc=False)
    try: 
        qdrant_client.get_collection(QDRANT_COLLECTION_NAME)
    except Exception:
        qdrant_client.recreate_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=models.VectorParams(size=len(embedding.embed_query("Test text")), distance=models.Distance.COSINE),
        )
        logger.info(f'SUCCESS: Collection {QDRANT_COLLECTION_NAME} created.')
    vector_db = Qdrant(client=qdrant_client, collection_name=QDRANT_COLLECTION_NAME, embeddings=embedding)
    logger.info("SUCCESS: Qdrant DB Connection.")
    return vector_db