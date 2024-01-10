import weaviate
from weaviate.client import Client
from langchain.vectorstores.weaviate import Weaviate

from .base import VectorStore
from .embedding.base import BaseEmbedding


def create_weaviate_client(url: str, api_key: str) -> Client:
    weaviate_client = Client(
        url=url, 
        auth_client_secret=weaviate.AuthApiKey(api_key),
    )
    return weaviate_client



class WeaviateVectorStore(VectorStore):
    def __init__(
            self,
            client: Client,
            index_name: str,
            embedding_model: BaseEmbedding,
            text_field: str = "text",
    ):
        self.weaviate_client = client
        self.embedding_model = embedding_model
        self.index_name = index_name
        self.text_field = text_field

        self.client = Weaviate(
            client=client,
            index_name=index_name,
            embedding=embedding_model.client,
            text_key=text_field,
        )