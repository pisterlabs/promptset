import contextlib
from typing import List, Optional

from langchain.vectorstores.qdrant import Qdrant
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever

from aitestdrive.common.async_locks import ReadWriteLock
from aitestdrive.common.config import config


class QdrantService:
    __collection_name = 'aitestdrive-qdrant-collection'

    def __init__(self, embeddings: Optional[Embeddings] = None):
        self.__embeddings = embeddings
        self.__qdrant_store = Qdrant.construct_instance(url=config.qdrant_url,
                                                        api_key=config.qdrant_api_key,
                                                        embedding=embeddings,
                                                        collection_name=QdrantService.__collection_name,
                                                        texts=["this text only used for getting vector size"],
                                                        force_recreate=False)
        self.__lock = ReadWriteLock()

    @contextlib.asynccontextmanager
    async def read_context(self):
        async with self.__lock.reader():
            yield QdrantReadContext(self.__qdrant_store)

    async def re_upload_collection(self,
                                   documents: List[Document]):
        async with self.__lock.writer():
            await self.__qdrant_store.afrom_documents(documents=documents,
                                                      url=config.qdrant_url,
                                                      api_key=config.qdrant_api_key,
                                                      embedding=self.__embeddings,
                                                      collection_name=QdrantService.__collection_name,
                                                      force_recreate=True)


class QdrantReadContext:
    def __init__(self, qdrant_store):
        self.__qdrant_store: Qdrant = qdrant_store

    async def search(self, query: str, limit: int = 4) -> List[Document]:
        return await self.__qdrant_store.asimilarity_search(query, k=limit)

    def as_retriever(self, limit: int = 4) -> VectorStoreRetriever:
        return self.__qdrant_store.as_retriever(search_kwargs={'k': limit})
