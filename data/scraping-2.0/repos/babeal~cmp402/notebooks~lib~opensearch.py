import langchain.vectorstores.opensearch_vector_search as ovs

from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth, helpers
from langchain.vectorstores import OpenSearchVectorSearch


def create_ovs_client(
    collection_id,
    index_name,
    region,
    boto3_session,
    bedrock_embeddings,
):
    service = "aoss"
    host = f"{collection_id}.{region}.aoss.amazonaws.com"
    credentials = boto3_session.get_credentials()
    http_auth = AWSV4SignerAuth(credentials, region, service)

    aoss_runtime_client = OpenSearch(
        hosts=[{"host": host, "port": 443}],
        http_auth=http_auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=300,
        pool_maxsize=20,
    )

    patch_langchain(ovs, aoss_runtime_client)

    db = OpenSearchVectorSearch(
        opensearch_url=host,
        http_auth=http_auth,
        index_name=index_name,
        engine="nmslib",
        space_type="cosinesimil",
        embedding_function=bedrock_embeddings,
    )

    return db


def patch_langchain(ovs, aoss_runtime_client):
    def get_opensearch_client(opensearch_url: str, **kwargs):
        return aoss_runtime_client

    ovs._get_opensearch_client = get_opensearch_client
