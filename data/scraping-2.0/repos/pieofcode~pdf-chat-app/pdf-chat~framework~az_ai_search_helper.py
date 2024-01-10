import json
import os
import openai
from pathlib import Path
import dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient, SearchIndexingBufferedSender
from azure.search.documents.indexes import SearchIndexClient
from langchain.vectorstores.azuresearch import AzureSearch
from langchain.embeddings import AzureOpenAIEmbeddings
from azure.search.documents.indexes.models import (
    SemanticSettings,
    SemanticConfiguration,
    PrioritizedFields,
    SemanticField
)
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import TextSplitter

from azure.search.documents.models import *
from azure.search.documents.indexes.models import *


env_name = os.environ["APP_ENV"] if "APP_ENV" in os.environ else "local"

# Load env settings
env_file_path = Path(f"./.env.{env_name}")
print(f"Loading environment from: {env_file_path}")
with open(env_file_path) as f:
    dotenv.load_dotenv(dotenv_path=env_file_path)
# print(os.environ)


def get_az_search_index_client():
    endpoint = os.environ["AZURE_SEARCH_SERVICE_ENDPOINT"]
    key = os.environ["AZURE_SEARCH_ADMIN_KEY"]

    credential = AzureKeyCredential(key)

    index_client = SearchIndexClient(endpoint=endpoint, credential=credential)
    return index_client


def get_az_search_client():
    endpoint = os.environ["AZURE_SEARCH_SERVICE_ENDPOINT"]
    index_name = os.environ["AZURE_SEARCH_INDEX_NAME"]
    key = os.environ["AZURE_SEARCH_ADMIN_KEY"]

    credential = AzureKeyCredential(key)

    client = SearchClient(endpoint=endpoint,
                          index_name=index_name,
                          credential=credential)
    return client


def get_az_search_indices():

    service_client = get_az_search_index_client()
    # Get all the indices from the Azure Search service
    result = service_client.list_index_names()
    names = [x for x in result]
    # names = ["azure-plat-services-vector-search", "langchain-vector-demo"]
    return names


def get_index_fields(index_name, embedding_function):
    if index_name == "azure-plat-services-vector-search":
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String,
                        key=True, sortable=True, filterable=True, facetable=True),
            SearchableField(name="title", type=SearchFieldDataType.String),
            SearchableField(name="content", type=SearchFieldDataType.String),
            SearchableField(name="category", type=SearchFieldDataType.String,
                            filterable=True),
            SearchField(name="title_vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                        searchable=True, vector_search_dimensions=1536, vector_search_configuration="myHnswProfile"),
            SearchField(name="content_vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                        searchable=True, vector_search_dimensions=1536, vector_search_configuration="myHnswProfile"),
        ]

    else:
        fields = [
            SimpleField(
                name="id",
                type=SearchFieldDataType.String,
                key=True,
                filterable=True,
            ),
            SearchableField(
                name="content",
                type=SearchFieldDataType.String,
                searchable=True,
            ),
            SearchField(
                name="content_vector",
                type=SearchFieldDataType.Collection(
                    SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=len(embedding_function("Text")),
                vector_search_configuration="default",
            ),
            SearchableField(
                name="metadata",
                type=SearchFieldDataType.String,
                searchable=True,
            )
        ]

    return fields


def create_cogsearch_index(index_name, embeddings):

    print(f"Creating index: {index_name}")

    endpoint = os.environ["AZURE_SEARCH_SERVICE_ENDPOINT"]
    key = os.environ["AZURE_SEARCH_ADMIN_KEY"]

    vector_store_en: AzureSearch = AzureSearch(
        azure_search_endpoint=endpoint,
        azure_search_key=key,
        index_name=index_name,
        embedding_function=embeddings.embed_query,
        semantic_configuration_name='config',
        semantic_settings=SemanticSettings(
            default_configuration='config',
            configurations=[
                SemanticConfiguration(
                    name='config',
                    prioritized_fields=PrioritizedFields(
                        title_field=SemanticField(field_name='content'),
                        prioritized_content_fields=[
                            SemanticField(field_name='content')],
                        prioritized_keywords_fields=[
                            SemanticField(field_name='metadata')]
                    ))
            ])
    )


