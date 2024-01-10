from langchain.vectorstores import Chroma
from langchain.vectorstores.base import VectorStoreRetriever
from modules.vector_stores.loaders.pypdf_load_strategy import (
    get_default_pypdf_loader,
)
from modules.vector_stores.embedding.instructorxl import get_default_instructor_embedding

instruct_embed = get_default_instructor_embedding()
from dataclasses import dataclass
from langchain.embeddings.base import Embeddings
from typing import Iterable
from langchain.schema import Document
from loguru import logger


@dataclass
class ChromaConfig:
    documents: Iterable[Document]
    persist_directory: str
    embedding: Embeddings
    persisted: bool = False


class ChromaManager:
    def __init__(self, config: ChromaConfig):
        self.documents = config.documents
        self.persist_directory = config.persist_directory
        self.embedding = config.embedding
        if config.persisted:
            self.db = Chroma(
                persist_directory=config.persist_directory, embedding_function=config.embedding
            )
        else:
            self.db = Chroma.from_documents(
                documents=config.documents,
                embedding=config.embedding,
                persist_directory=config.persist_directory,
            )

    def persist(self):
        logger.info("Persisting Chroma to disk...")
        self.db.persist()
        logger.info("Chroma saved to %s", self.persist_directory)

    def delete(self):
        logger.info("Deleting Chroma from disk...")
        self.db.delete_collection()
        self.db.persist()
        logger.info("Chroma deleted from %s", self.persist_directory)

    def fetch_documents(self, query):
        logger.info("Fetching documents from Chroma...")
        retriever: VectorStoreRetriever = self.db.as_retriever()
        documents = retriever.get_relevant_documents(query)
        logger.info("Fetched %s documents from Chroma", len(documents))
        return documents


def get_default_chroma_mgr(persisted=False):
    """
    Returns a default ChromaConfig instance. The default currently only reads in pdf files from the data directory.

    Returns:
        ChromaConfig: A new ChromaConfig instance.
    """
    dir_location = "../data"
    persist_directory = "../db"
    loader = get_default_pypdf_loader(dir_location)
    documents: Iterable[Document] = loader.load()
    embedding = get_default_instructor_embedding()
    if persisted:
        config = ChromaConfig(
            documents=documents,
            persist_directory=persist_directory,
            embedding=embedding,
            persisted=True,
        )
    else:
        config = ChromaConfig(
            documents=documents, persist_directory=persist_directory, embedding=embedding
        )
    chroma_mgr = ChromaManager(config)
    return chroma_mgr

