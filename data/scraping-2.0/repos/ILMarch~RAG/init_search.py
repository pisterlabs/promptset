"""
Инициализирует индекс Azure Cognitive Search нашими данными, используя векторный поиск
и семантическое ранжирование.

Для запуска этого кода в вашей учётной записи Azure уже должны быть созданы ресурсы
"Cognitive Search" и "OpenAI".
"""
import os

import openai
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswParameters,
    HnswVectorSearchAlgorithmConfiguration,
    PrioritizedFields,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SemanticConfiguration,
    SemanticField,
    SemanticSettings,
    SimpleField,
    VectorSearch,
)
from dotenv import load_dotenv
from langchain.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter

# Конфигурация Azure Search.
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX_NAME = "products-index-1"

# Конфигурация Azure OpenAI.
AZURE_OPENAI_API_TYPE = "azure"
AZURE_OPENAI_API_BASE = os.getenv("AZURE_OPENAI_API_BASE")
AZURE_OPENAI_API_VERSION = "2023-03-15-preview"
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

DATA_DIR = "data/"


def load_and_split_documents() -> list[dict]:
    """
    Загружает наши документы с диска и разбивает их на фрагменты.
    Возвращает список словарей.
    """
    # Загрузка данных.
    loader = DirectoryLoader(
        DATA_DIR, loader_cls=UnstructuredMarkdownLoader, show_progress=True
    )
    docs = loader.load()

    # Разбиение документов на фрагменты.
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.MARKDOWN, chunk_size=6000, chunk_overlap=100
    )
    split_docs = splitter.split_documents(docs)

    # Преобразование документов в список словарей.
    final_docs = []
    for i, doc in enumerate(split_docs):
        doc_dict = {
            "id": str(i),
            "content": doc.page_content,
            "sourcefile": os.path.basename(doc.metadata["source"]),
        }
        final_docs.append(doc_dict)

    return final_docs


def get_index(name: str) -> SearchIndex:
    """
    Возвращает индекс Azure Cognitive Search с заданным именем.
    """
    # Поля, которые мы хотим индексировать. Поле "embedding" - это векторное поле,
    # которое будет использоваться для векторного поиска.
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SimpleField(name="sourcefile", type=SearchFieldDataType.String),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SearchField(
            name="embedding",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            # Размер вектора, созданного моделью text-embedding-ada-002.
            vector_search_dimensions=1536,
            vector_search_configuration="default",
        ),
    ]

    # Поле "content" должно иметь приоритет при семантическом ранжировании.
    semantic_settings = SemanticSettings(
        configurations=[
            SemanticConfiguration(
                name="default",
                prioritized_fields=PrioritizedFields(
                    title_field=None,
                    prioritized_content_fields=[SemanticField(field_name="content")],
                ),
            )
        ]
    )

    # Для векторного поиска мы хотим пользоваться алгоритмом HNSW (Hierarchical Navigable Small World)
    # (разновидность алгоритма приближённого поиска ближайшего соседа) с
    # применением косинусного коэффициента векторов.
    vector_search = VectorSearch(
        algorithm_configurations=[
            HnswVectorSearchAlgorithmConfiguration(
                name="default",
                kind="hnsw",
                parameters=HnswParameters(metric="cosine"),
            )
        ]
    )

    # Создание поискового индекса.
    index = SearchIndex(
        name=name,
        fields=fields,
        semantic_settings=semantic_settings,
        vector_search=vector_search,
    )

    return index


def initialize(search_index_client: SearchIndexClient):
    """
    Инициализирует индекс Azure Cognitive Search нашими данными, используя
векторный поиск.
    """
    # Загрузка данных.
    docs = load_and_split_documents()
    for doc in docs:
        doc["embedding"] = openai.Embedding.create(
            engine=AZURE_OPENAI_EMBEDDING_DEPLOYMENT, input=doc["content"]
        )["data"][0]["embedding"]

    # Создание индекса Azure Cognitive Search.
    index = get_index(AZURE_SEARCH_INDEX_NAME)
    search_index_client.create_or_update_index(index)

    # Выгрузка данных в индекс.
    search_client = SearchClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        index_name=AZURE_SEARCH_INDEX_NAME,
        credential=AzureKeyCredential(AZURE_SEARCH_KEY),
    )
    search_client.upload_documents(docs)


def delete(search_index_client: SearchIndexClient):
    """
    Удаляет индекс Azure Cognitive Search.
    """
    search_index_client.delete_index(AZURE_SEARCH_INDEX_NAME)


def main():
    load_dotenv()

    openai.api_type = AZURE_OPENAI_API_TYPE
    openai.api_base = AZURE_OPENAI_API_BASE
    openai.api_version = AZURE_OPENAI_API_VERSION
    openai.api_key = AZURE_OPENAI_API_KEY

    search_index_client = SearchIndexClient(
        AZURE_SEARCH_ENDPOINT, AzureKeyCredential(AZURE_SEARCH_KEY)
    )

    initialize(search_index_client)
    # delete(search_index_client)


if __name__ == "__main__":
    main()