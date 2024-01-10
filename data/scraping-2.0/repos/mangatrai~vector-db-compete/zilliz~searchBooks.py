import csv
import json
import random
import openai
import time
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

# Searching book titles

# Extract embedding from text using OpenAI
def embed(text):
    return openai.Embedding.create(
        input=text,
        engine=OPENAI_ENGINE)["data"][0]["embedding"]

# Define parameters
FILE = 'books.csv'  # Download it from https://www.kaggle.com/datasets/jealousleopard/goodreadsbooks and save it in the folder that holds your script.
COLLECTION_NAME = 'title_db'  # Collection name
DIMENSION = 1536  # Embeddings size
COUNT = 200  # How many titles to embed and insert
URI = ''  # Endpoint URI obtained from Zilliz Cloud
USER = ''  # Username specified when you created this cluster
PASSWORD=''  # Password set for that account
OPENAI_ENGINE = 'text-embedding-ada-002'  # Which engine to use
openai.api_key = ''  # Use your own Open AI API Key here
zilliz_api_key = '' # Use your own Zilliz Cloud API key here

# For serverless cluster
TOKEN = zilliz_api_key

# For dedicated cluster
# TOKEN = f"{USER}:{PASSWORD}"

# Connect to Zilliz Cloud
connections.connect(uri=URI, token=TOKEN, secure=True)

collection = Collection(COLLECTION_NAME)
collection.load()

# Search the cluster based on input text
def search(text):
    # Search parameters for the index
    search_params={
        "metric_type": "L2"
    }

    results=collection.search(
        data=[embed(text)],  # Embeded search value
        anns_field="embedding",  # Search across embeddings
        param=search_params,
        limit=5,  # Limit to five results per search
        output_fields=['title']  # Include title field in result
    )

    ret=[]
    for hit in results[0]:
        row=[]
        row.extend([hit.id, hit.score, hit.entity.get('title')])  # Get the id, distance, and title for the results
        ret.append(row)
    return ret

search_terms=['self-improvement', 'landscape', 'computer']

for x in search_terms:
    print('Search term:', x)
    for result in search(x):
        print(result)
    print()
