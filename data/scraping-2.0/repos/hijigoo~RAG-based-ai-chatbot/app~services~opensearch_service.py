import boto3

from langchain.vectorstores import OpenSearchVectorSearch
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
from langchain.embeddings.base import Embeddings

region = 'us-west-2'
# endpoint_url = 'https://search-doc-vector-store-d6ewfi4eflxfciyyticvh5zm5m.us-west-2.es.amazonaws.com'
endpoint_url = 'https://vpc-doc-vector-store-vpc-44f2zwlvjspbifxgg33tf74dou.us-west-2.es.amazonaws.com'

service = 'es'  # must set the service as 'es'

# 권한이 있는 Task Role 을 ECS Task 에 등록 하고, OpenSearch 에 Task Role 을 맵핑해서 사용
credentials = boto3.Session().get_credentials()
awsauth = AWS4Auth(
    region=region,
    service=service,
    refreshable_credentials=credentials)


def get_opensearch_client():
    return OpenSearch(
        region=region,
        hosts=[{'host': endpoint_url.replace("https://", ""), 'port': 443}],
        http_auth=awsauth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection
    )


def check_if_index_exists(index_name: str) -> bool:
    os_client = get_opensearch_client()
    exists = os_client.indices.exists(index_name)
    return exists


def create_index(index_name: str):
    os_client = get_opensearch_client()
    os_client.indices.create(index=index_name)


def delete_index(index_name: str):
    os_client = get_opensearch_client()
    return os_client.indices.delete(index=index_name)


def get_index_list(index_name: str):
    os_client = get_opensearch_client()
    return os_client.indices.get_alias(index=index_name)


def create_index_from_documents(index_name: str, embeddings: Embeddings, documents):
    return OpenSearchVectorSearch.from_documents(
        documents=documents,
        embedding=embeddings,
        opensearch_url=endpoint_url,
        timeout=300,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        http_auth=awsauth,
        index_name=index_name
    )


def get_opensearch_vector_client(index_name: str, embeddings: Embeddings):
    return OpenSearchVectorSearch(
        opensearch_url=endpoint_url,
        index_name=index_name,
        embedding_function=embeddings,
        is_aoss=False,
        connection_class=RequestsHttpConnection,
        http_auth=awsauth,
    )


def get_most_similar_docs_by_query(index_name: str, embeddings: Embeddings, query: str, k: int):
    osv_client = get_opensearch_vector_client(index_name, embeddings)
    return osv_client.similarity_search(
        query,
        k=k
    )
