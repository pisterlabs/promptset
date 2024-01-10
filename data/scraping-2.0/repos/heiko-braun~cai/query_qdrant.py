from openai import OpenAI
from conf.constants import *

from qdrant_client import QdrantClient
import argparse

# ---


# create an embedding using openai
def get_embedding(openai_client, text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   resp = openai_client.embeddings.create(input = [text], model=model)
   return resp.data[0].embedding

# query the vector store
def query_qdrant(openai_client, qdrant_client, query, collection_name, top_k=5):
    
    embedded_query = get_embedding(openai_client=openai_client, text=query)
    
    query_results = qdrant_client.search(
        collection_name=collection_name,
        query_vector=(embedded_query),
        limit=top_k,
    )
    
    return query_results

# ---

# OpenAI Client
openai_client = OpenAI()

# Vector DB
qdrant_client = QdrantClient(
    QDRANT_URL,
    api_key=QDRANT_KEY,
)

# arguments
parser = argparse.ArgumentParser(description='Extract PDF pages')
parser.add_argument('-c', '--collection', help='The target collection name', required=True)
parser.add_argument('-k', '--topk', help='Num top k', required=False, default=5)
args = parser.parse_args()

# exceute query
query_results = query_qdrant(
    openai_client=openai_client, 
    qdrant_client=qdrant_client, 
    query=input("Prompt:"), 
    collection_name=args.collection,
    top_k=args.topk
    )

# list results oreder by score
print("Found N matches: ", len(query_results))
for i, article in enumerate(query_results):    
    #print(article)
    print(f'{i + 1}. {article.payload["metadata"]["page_number"]} (Score: {round(article.score, 3)})')

