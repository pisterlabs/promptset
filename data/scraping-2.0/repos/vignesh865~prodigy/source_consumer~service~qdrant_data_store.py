from typing import List

from django.conf import settings
from langchain.schema import Document
from langchain.schema.embeddings import Embeddings
from langchain.vectorstores.qdrant import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http import models

from source_consumer.service.data_store import DataStore


class QdrantDataStore(DataStore):

    def __init__(self, embedding_model: Embeddings, vector_size: int):
        self.embedding_model = embedding_model

        db_config = settings.QDRANT_DB_CONFIG
        self.qdrant_client = QdrantClient(host=db_config["host"],
                                          port=db_config["port"],
                                          api_key=db_config["api_key"])

        self.vector_size = vector_size

    def get_store(self, collection_name):
        return Qdrant(self.qdrant_client, collection_name, self.embedding_model)

    def update_data(self, collection_name: str, chunked_docs: List[Document]):
        self.create_collection(collection_name)
        qdrant = self.get_store(collection_name)
        return qdrant.add_documents(chunked_docs)

    def create_collection(self, collection_name):
        if self.qdrant_client.get_collection(collection_name):
            return

        self.qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=self.vector_size, distance=models.Distance.COSINE),
        )

    def get_documents(self, collection_name):

        lang_documents = []
        documents = self.qdrant_client.scroll(
            collection_name=collection_name,
            limit=1000,
            with_payload=True,
            with_vectors=False,
        )[0]

        for document in documents:
            print(document)
            lang_documents.append(
                Document(metadata=document.payload.get('metadata'),
                         page_content=document.payload.get('page_content')))

        return lang_documents
