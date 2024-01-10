from typing import TYPE_CHECKING

from langchain.vectorstores.weaviate import Weaviate
from weaviate import Client as weaviateClient

from gpt_context.adapters.vector_store_adapter import VectorStoreAdapter
from gpt_context.exceptions.business_logic_exception import BusinessLogicException
from gpt_context.exceptions.errors import ErrorCode, ErrorMessage

if TYPE_CHECKING:
    from gpt_context.services import AppConfig


class WeaviateVectorStoreAdapter(VectorStoreAdapter):
    def __init__(
            self,
            config: 'AppConfig',
            text_key="text",
            embedding=None,
            attributes=["source"]
    ):
        self.url = config.WEAVIATE_URL
        self.text_key = text_key
        self.embedding = embedding
        self.attributes = attributes

        self._client = weaviateClient(self.url)

    def get_retriever(self, index_name: str):
        weaviate = self._init_index(index_name)

        return weaviate.as_retriever()

    def from_documents(self, documents, index_name, **kwargs):
        weaviate = self._init_index(index_name)

        weaviate.from_documents(documents, self.embedding, index_name=index_name)

    def truncate(self, context_name: str):
        if not self.context_exists(context_name):
            raise BusinessLogicException(
                ErrorCode.NOT_FOUND,
                ErrorMessage.CONTEXT_NOT_FOUND.format(context_name=context_name)
            )

        query_result = self._client.query.get(context_name).with_additional(["id"]).do()
        ids = list(map(lambda x: x["_additional"]["id"], query_result["data"]["Get"][context_name]))

        for uuid in ids:
            self._client.data_object.delete(uuid, context_name)

    def context_exists(self, context_name: str) -> bool:
        return self._client.schema.exists(context_name)

    def _init_index(self, index_name: str) -> Weaviate:
        return Weaviate(
            self._client,
            index_name,
            self.text_key,
            by_text=False,
            embedding=self.embedding,
            attributes=self.attributes
        )
