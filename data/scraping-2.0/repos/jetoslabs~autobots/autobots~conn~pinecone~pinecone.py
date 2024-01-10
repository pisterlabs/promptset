import time
from functools import lru_cache
from typing import List

from pinecone import Client, Index, QueryResult
# import pinecone
# from pinecone import QueryResponse, FetchResponse, ScoredVector
# from pinecone.core.grpc.protos.vector_service_pb2 import DeleteResponse

from autobots.conn.openai.openai_embeddings.embedding_model import EmbeddingReq, EmbeddingRes
from autobots.conn.openai.openai_client import OpenAI, get_openai
from autobots.core.logging.log import Log
from autobots.core.settings import Settings, SettingsProvider


class Pinecone:

    def __init__(
            self,
            api_key: str,
            environment: str,
            open_ai: OpenAI = get_openai(),
            index_name: str = "index-1536",
            dimension: int = 1536,
    ):
        if not api_key or not environment:
            return
        self.open_ai = open_ai
        self.pinecone = Client(api_key=api_key, region=environment)
        self.index_name = index_name
        self.dimension = dimension
        self.index = self.create_index()

    def create_index(self) -> Index:
        """
        Only create index if it doesn't exist
        :return:
        """
        if self.index_name not in self.pinecone.list_indexes():
            self.pinecone.create_index(name=self.index_name, dimension=self.dimension)
            time.sleep(3)
        index = self.pinecone.Index(self.index_name)
        return index

    async def upsert_data(self, vector_id: str, data: str, metadata: dict, namespace: str = "default"):
        try:
            upserted = []
            embedding_req = EmbeddingReq(input=data)
            embedding_res: EmbeddingRes = await self.open_ai.openai_embeddings.embeddings(embedding_req)
            for embedding_data in embedding_res.data:
                # vector = (Vector_ID, Dense_vector_values, Vector_metadata)
                vector = (vector_id, embedding_data.embedding, metadata)
                upsert_res = self.index.upsert(vectors=[vector], namespace=namespace)
                upserted.append(upsert_res)
        except Exception as e:
            Log.error(str(e))

    async def query(
            self,
            data: str,
            namespace: str = "default",
            top_k: int = 10,
            filter: dict = None,
            include_values: bool = True,
            include_metadata: bool = True,
    ) -> List[QueryResult]:
        embedding_req = EmbeddingReq(input=data)
        embedding_res: EmbeddingRes = await self.open_ai.openai_embeddings.embeddings(embedding_req)
        try:
            for embedding_data in embedding_res.data:
                res: List[QueryResult] = self.index.query(
                    values=embedding_data.embedding,
                    namespace=namespace,
                    top_k=top_k,
                    filter=filter,
                    include_values=include_values,
                    include_metadata=include_metadata
                )
                return res
        except Exception as e:
            Log.error(str(e))

    async def fetch(self, vector_ids: List[str], namespace: str = "default") -> dict:
        fetch_res = self.index.fetch(ids=vector_ids, namespace=namespace)
        return fetch_res

    async def delete_all(self, namespace: str | None = None):
        deleted = self.index.delete_all(namespace=namespace)
        return deleted

    async def delete_metadata(
            self,
            filter: dict[str, str | float | int | bool | list | dict] | None = None,
            namespace: str | None = None
    ) -> dict:
        deleted = self.index.delete_by_metadata(filter=filter, namespace=namespace)
        return deleted


@lru_cache
def get_pinecone(settings: Settings = SettingsProvider.sget()) -> Pinecone:
    return Pinecone(api_key=settings.PINECONE_API_KEY, environment=settings.PINECONE_ENVIRONMENT)
