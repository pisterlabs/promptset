from typing import Optional, Union
import chromadb
from chromadb.api import API
from chromadb.api.models.Collection import Collection
from chromadb.api.types import EmbeddingFunction
from chromadb.utils import embedding_functions
from pydantic import BaseModel
from chat.common.const import PERSIST_DIRECTORY, EMBEDDING_MODEL
from chat.common.config import OPENAI_API_KEY
from chat.common.log import logger


class EmbeddingResult(BaseModel):
    ids: list
    embeddings: list[list[float]]
    metadatas: Optional[list]
    documents: list[str]

    @property
    def size(self):
        return len(self.embeddings)


class Chroma:
    """ chromadb wrapper. Only supports persistent client for now. """
    persist_directory: str
    client: API
    embedding_function: EmbeddingFunction
    collections: dict[str, Collection]

    def __init__(self):
        self.persist_directory = PERSIST_DIRECTORY
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=OPENAI_API_KEY,
            model_name=EMBEDDING_MODEL,
        )

    def create_collection(
            self,
            name: str,
            metadata=None
    ) -> Collection:
        return self.client.create_collection(
            name=name,
            metadata=metadata,
            embedding_function=self.embedding_function,
        )

    def get_collection(
            self,
            name: str
    ) -> Collection:
        return self.client.get_collection(
            name,
            embedding_function=self.embedding_function,
        )

    def get_or_create_collection(
            self,
            name: str,
            metadata=None
    ) -> Collection:
        return self.client.get_or_create_collection(
            name=name,
            metadata=metadata,
            embedding_function=self.embedding_function,
        )

    def add(
            self,
            name: str,
            ids: list[str],
            documents: list[str],
            metadatas=None,
    ):
        self.get_collection(name).add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

    def query(
            self,
            name: str,
            query_texts: Union[str, list[str]],
            n_results: int = 2,
            **kwargs,
    ):
        logger.debug('using collection `%s` to query result for %s', name, query_texts)
        return self.get_collection(name).query(
            query_texts=query_texts,
            n_results=n_results,
            **kwargs,
        )
