"""Vectordatabase Utlilty Functions."""
import os

from dotenv import load_dotenv
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Qdrant
from loguru import logger
from omegaconf import DictConfig
from qdrant_client import QdrantClient, models

from ollama_rag.utils.configuration import load_config

load_dotenv()


@load_config(location="config/main.yml")
def get_db_connection(cfg: DictConfig, collection_name: str) -> Qdrant:
    """Initializes a connection to the Qdrant DB.

    Args:
        cfg (DictConfig): The configuration file loaded via OmegaConf.
        aleph_alpha_token (str): The Aleph Alpha API token.

    Returns:
        Qdrant: The Qdrant DB connection.
    """
    embedding = OllamaEmbeddings(base_url=cfg.ollama_embeddings.url, model=cfg.ollama_embeddings.model)
    qdrant_client = QdrantClient(
        cfg.qdrant.url,
        port=cfg.qdrant.port,
        api_key=os.getenv("QDRANT_API_KEY"),
        prefer_grpc=cfg.qdrant.prefer_grpc,
    )
    if collection_name is None or collection_name == "":
        collection_name = cfg.qdrant.collection_name_ollama
    vector_db = Qdrant(client=qdrant_client, collection_name=collection_name, embeddings=embedding)
    logger.info("SUCCESS: Qdrant DB initialized.")

    return vector_db


@load_config(location="config/main.yml")
def initialize_qdrant_client_config(cfg: DictConfig):
    """Initialize the Qdrant Client.

    Args:
        cfg (DictConfig): Configuration from the file

    Returns:
        _type_: Qdrant Client and Configuration.
    """
    qdrant_client = QdrantClient(
        cfg.qdrant.url,
        port=cfg.qdrant.port,
        api_key=os.getenv("QDRANT_API_KEY"),
        prefer_grpc=cfg.qdrant.prefer_grpc,
    )
    return qdrant_client, cfg


def initialize_ollama_vector_db() -> None:
    """Initializes the GPT4ALL vector db.

    Args:
        cfg (DictConfig): Configuration from the file
    """
    qdrant_client, cfg = initialize_qdrant_client_config()

    try:
        qdrant_client.get_collection(collection_name=cfg.qdrant.collection_name_ollama)
        logger.info(f"SUCCESS: Collection {cfg.qdrant.collection_name_ollama} already exists.")
    except Exception:
        generate_collection_ollama(qdrant_client, collection_name=cfg.qdrant.collection_name_ollama)


def generate_collection_ollama(qdrant_client, collection_name):
    """Generate a collection for the GPT4ALL Backend.

    Args:
        qdrant_client (Qdrant): Qdrant Client
        collection_name (str): Name of the Collection
    """
    qdrant_client.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=4096, distance=models.Distance.COSINE),
    )
    logger.info(f"SUCCESS: Collection {collection_name} created.")
