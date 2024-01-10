import openai
from opensearchpy import OpenSearch
import pandas as pd

# Initialize OpenAI and OpenSearch clients
openai.api_key = 'get_your_own_key'


def get_embedding(text, model="text-embedding-ada-002"):
    """
    Use the OpenAI model to generate embeddings for a given text.
    Returns the embeddings.
    """
    text = text.replace("\n", " ")  # Replace newlines with spaces
    return openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']


embedding = get_embedding(text='My Name is Manu')
print(embedding)

# Initialize OpenSearch client

opensearch_client = OpenSearch(
    hosts=['https://localhost:9200'],
    http_auth=('admin', 'admin'),
    use_ssl=True,
    verify_certs=False,  # Only for development! Do not use in production.
    ssl_show_warn=False  # To suppress SSL-related warnings
)

# To test the connection:
print(opensearch_client.info())

settings = {
    "mappings": {
        "properties": {
            "my_vector": {
                "type": "nested",
                "properties": {
                    "value": {"type": "float"}
                }
            },
            "name": {
                "type": "text"
            }
        }
    }
}


def create_opensearch_index(index_name):
    index_exists = opensearch_client.indices.exists(index_name)
    if not index_exists:
        # Create index if it doesn't exist
        opensearch_client.indices.create(index=index_name, body=settings)


def open_search_import(index_name, df):
    create_opensearch_index(index_name)
    for _, row in df.iterrows():
        embedding = get_embedding(row['text'])
        document = {
            "name": row['text'],
            "my_vector": [{"value": v} for v in embedding]
        }
        # opensearch_client.index(index=index_name, body=document)
        doc_id = str(row['unique_id'])  # assuming you have a column named 'unique_id' as the unique identifier
        opensearch_client.index(index=index_name, id=doc_id, body=document)


df = pd.read_csv('manu.csv')
print(df)
index_name = 'manu_index_1'
open_search_import(index_name, df)
