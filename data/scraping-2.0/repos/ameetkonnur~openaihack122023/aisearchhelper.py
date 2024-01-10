from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
import os
from langchain.document_loaders import PyPDFLoader
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    VectorSearchProfile,
    HnswAlgorithmConfiguration,
)
from azure.search.documents.models import VectorizedQuery
import openai

service_endpoint = os.environ["AZURE_SEARCH_SERVICE_ENDPOINT"]
# index_name = os.environ["AZURE_SEARCH_INDEX_NAME"]
index_name = "vector-docs-index" # default name if not set
key = os.environ["AZURE_SEARCH_API_KEY"]
open_ai_endpoint = os.getenv("OPENAI_API_URL")
open_ai_key = os.getenv("OPENAI_API_KEY")
credential = AzureKeyCredential(key)

def set_index_name(name: str):
    index_name = name

def list_documents_in_index():
    search_client = SearchClient(service_endpoint, index_name, AzureKeyCredential(key))

    result = search_client.get_document_count()

    print("There are {} documents in the {} search index.".format(result, index_name))

def get_embeddings(text: str):

    client = openai.AzureOpenAI(
        azure_endpoint=open_ai_endpoint,
        api_key=open_ai_key,
        api_version="2023-09-01-preview",
    )
    embedding = client.embeddings.create(input=[text], model="ada-002")
    return embedding.data[0].embedding


def get_documents_index(name: str):

    fields = [
        SearchableField(name="chunkId", type=SearchFieldDataType.String, key=True, analyzer_name="keyword"),
        SearchableField(name="documentTitle", type=SearchFieldDataType.String, sortable=True, filterable=True, facetable=True, analyzer_name="keyword"),
        SearchableField(name="documentContent", type=SearchFieldDataType.String),
        SearchField(
            name="contentVector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=1536,
            vector_search_profile_name="my-vector-config",
        ),
        SearchableField(
            name="owner",
            type=SearchFieldDataType.String,
            sortable=True,
            filterable=True,
            facetable=True,
        ),
    ]
    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="my-algorithms-config")],
        profiles=[VectorSearchProfile(name="my-vector-config", algorithm_configuration_name="my-algorithms-config")]
    )
    return SearchIndex(name=name, fields=fields, vector_search=vector_search)

def single_vector_search(query):
    # [START single_vector_search]

    search_client = SearchClient(service_endpoint, index_name, AzureKeyCredential(key))
    vector_query = VectorizedQuery(vector=get_embeddings(query), k_nearest_neighbors=1, fields="contentVector")

    results = search_client.search(
        vector_queries=[vector_query],
        select=["documentTitle", "documentContent"],
    )

    search_text = ""
    for result in results:
        search_text = search_text + result["documentContent"]
    return search_text
    # [END single_vector_search]


def single_vector_search_with_filter(query,owner):
    # [START single_vector_search_with_filter]

    search_client = SearchClient(service_endpoint, index_name, AzureKeyCredential(key))
    vector_query = VectorizedQuery(vector=get_embeddings(query), k_nearest_neighbors=1, fields="contentVector")

    results = search_client.search(
        search_text="",
        vector_queries=[vector_query],
        filter="owner eq '" + owner + "'",
        select=["documentTitle", "documentContent"],
    )

    search_text = ""
    for result in results:
        search_text = search_text + result["documentContent"]
    return search_text
    # [END single_vector_search_with_filter]


def simple_hybrid_search(query):
    search_client = SearchClient(service_endpoint, index_name, AzureKeyCredential(key))
    vector_query = VectorizedQuery(vector=get_embeddings(query), k_nearest_neighbors=3, fields="contentVector")

    results = search_client.search(
        search_text=query,
        vector_queries=[vector_query],
        select=["documentTitle", "documentContent"],
    )
    print(results.get_answers())
    for result in results:
        print(result)
    # [END simple_hybrid_search]

def get_documents(filename,owner):
    import uuid
    random_uid = str(uuid.uuid4())
    loader = PyPDFLoader(filename)
    pages = loader.load_and_split()
    docs = []

    # set default owner as ameetk
    if not owner:
        owner

    for page in pages:
        docs.append({
            "chunkId": random_uid + "__" + str(page.metadata["page"]),
            "documentTitle": filename,
            "documentContent": page.page_content,
            "contentVector": get_embeddings(page.page_content),
            "owner": owner,
        })
    return docs

def create_index(name: str):
    index_name = name
    index_client = SearchIndexClient(service_endpoint, credential)
    index = get_documents_index(index_name)
    index_client.create_index(index)

def upload_document(filename,owner):
    search_client = SearchClient(service_endpoint, index_name, credential)
    docs = get_documents(filename,owner)
    result = search_client.upload_documents(docs)
    if result[0].succeeded:
        print("Upload of documents succeeded.")
    else:
        print("Upload of documents failed.")
        print(result[0].errorMessage)