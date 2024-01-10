import os

from langchain.vectorstores.azuresearch import AzureSearch
from azure.search.documents.indexes.models import (
    SemanticSettings,
    SemanticConfiguration,
    PrioritizedFields,
    SemanticField,
)

VECTOR_STORE_ADDRESS = os.environ.get("VECTOR_STORE_ADDRESS")
VECTOR_STORE_PASSWORD = os.environ.get("VECTOR_STORE_PASSWORD")


class AzureVectorStorage:
    _instance = None

    def __new__(cls, index_name, embeddings):
        if cls._instance is None:
            cls._instance = super(AzureVectorStorage, cls).__new__(cls)
            cls._instance.vector_storage = AzureSearch(
                azure_search_endpoint=VECTOR_STORE_ADDRESS,
                azure_search_key=VECTOR_STORE_PASSWORD,
                index_name=index_name,
                embedding_function=embeddings.embed_query,
                semantic_configuration_name="config",
                semantic_settings=SemanticSettings(
                    default_configuration="config",
                    configurations=[
                        SemanticConfiguration(
                            name="config",
                            prioritized_fields=PrioritizedFields(
                                title_field=SemanticField(field_name="content"),
                                prioritized_content_fields=[
                                    SemanticField(field_name="content")
                                ],
                                prioritized_keywords_fields=[
                                    SemanticField(field_name="metadata")
                                ],
                            ),
                        )
                    ],
                ),
            )
        return cls._instance

    async def add_documents(self, docs):
        ids = self.vector_storage.add_documents(documents=docs)
        return ids

    def search(self, query):
        docs = self.vector_storage.similarity_search(
            query=query,
            k=1,
            search_type="hybrid",
        )
        return docs
