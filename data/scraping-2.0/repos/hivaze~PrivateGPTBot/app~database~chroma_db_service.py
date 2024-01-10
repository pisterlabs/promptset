import logging
from typing import List

import chromadb
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.vectorstores import Chroma

from app import settings

logger = logging.getLogger(__name__)

# host needs to be changed in prod to chroma_server (as in docker-compose.yml), for local test - localhost
chroma_client: chromadb.HttpClient = chromadb.HttpClient(host='chroma_server', port="8000")
chroma_client.heartbeat()

embeddings = OpenAIEmbeddings(openai_api_key=settings.config.OPENAI_KEY,
                              model=settings.config.embeddings_model.model_name,
                              embedding_ctx_length=settings.config.embeddings_model.embedding_ctx_length,
                              max_retries=settings.config.embeddings_model.max_retries)


def delete_if_exists(collection_name) -> bool:
    collections = set([x.name for x in chroma_client.list_collections()])
    if collection_name in collections:
        chroma_client.delete_collection(collection_name)
        logger.debug(f"Chroma collection '{collection_name}' deleted.")
        return True
    return False


def create_vector_store(user_id: int) -> Chroma:
    collection_name = str(user_id)
    delete_if_exists(collection_name)
    return Chroma(collection_name=collection_name, embedding_function=embeddings, client=chroma_client)


def add_documents(vectorstore: Chroma, documents: List[Document]):
    return vectorstore.add_documents(documents=documents)
