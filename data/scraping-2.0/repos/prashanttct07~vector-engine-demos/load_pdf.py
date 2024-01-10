
########
import requests
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import boto3
import json
import os
import numpy as np
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from sentence_transformers import SentenceTransformer


import numpy as np
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, PyPDFDirectoryLoader


# Load the SentenceTransformer model
model_name = 'sentence-transformers/msmarco-distilbert-base-tas-b'
model = SentenceTransformer(model_name)

# Set the desired vector size
vector_size = 768

# create open search collection public endpoint in us-east-2
AWS_PROFILE = "273117053997-us-east-2"
host = '0n2qav61946ja1c7k2a1.us-east-2.aoss.amazonaws.com' # OpenSearch Serverless collection endpoint
region = 'us-east-2' # e.g. us-west-2
opensearch_index = "garther-demo"

service = 'aoss'
credentials = boto3.Session(profile_name=AWS_PROFILE).get_credentials()
awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, service,
session_token=credentials.token)

# Create an OpenSearch client
client = OpenSearch(
    hosts = [{'host': host, 'port': 443}],
    http_auth = awsauth,
    timeout = 300,
    use_ssl = True,
    verify_certs = True,
    connection_class = RequestsHttpConnection
)

def index_embedding():
    actions =[]
    bulk_size = 0
    action = {"index": {"_index": opensearch_index}}
    
    loader = PyPDFDirectoryLoader("./data/")
    documents = loader.load()
    # - in our testing Character split works better with this PDF data set
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = 1000,
        chunk_overlap  = 100,
    )
    docs = text_splitter.split_documents(documents)

    for doc in docs: 
        # sample_embedding = np.array(bedrock_embeddings.embed_query(document.page_content))
        embeddings = model.encode(doc.page_content)
        vector_document = {
            "content": doc.page_content,
            "v_content": embeddings
        }
        # print (vector_document)
        actions.append(action)
        actions.append(vector_document)

        bulk_size+=1
        if(bulk_size > 100 ):
            client.bulk(body=actions)
            print(f"bulk request sent with size: {bulk_size}")
            bulk_size = 0

    #ingest remaining documents
    print("Sending remaining documents with size: ", bulk_size)
    client.bulk(body=actions)

def bootstrap_index():
    # create a new index
    index_body = {
        "settings": {
            "index.knn": True
        },
        'mappings': {
            'properties': {
                "content": { "type": "text", "fields": { "keyword": { "type": "keyword" } } }, #the field will be title.keyword and the data type will be keyword, this will act as sub field for
                "v_content": { "type": "knn_vector", "dimension": vector_size },
            }
        }
    }

    client.indices.create(
        index=opensearch_index, 
        body=index_body
    )

    client.indices.get_mapping(opensearch_index)    

        
# bootstrap_index()
index_embedding()
