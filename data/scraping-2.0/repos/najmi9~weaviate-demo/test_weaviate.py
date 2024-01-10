import weaviate
from dotenv import load_dotenv
import os
from langchain.embeddings import HuggingFaceEmbeddings
import json

load_dotenv()

WEAVIATE_HOST = os.getenv('WEAVIATE_HOST')
WEAVIATE_SECRET_KEY = os.getenv('WEAVIATE_SECRET_KEY')

client = weaviate.Client(
    url = WEAVIATE_HOST,
    auth_client_secret=weaviate.AuthApiKey(api_key=WEAVIATE_SECRET_KEY)
)

model_name = 'distiluse-base-multilingual-cased'
emb=HuggingFaceEmbeddings(model_name=model_name)

def query(input_text, k):
    input_embedding = emb.embed_query(input_text)
    vec = {"vector": input_embedding}
    result = client \
        .query.get("Text", ["ogr", "context", "libelle"]) \
        .with_near_vector(vec) \
        .with_limit(k) \
        .do()
    return result

input_text = "d'engins forestiers"
k_vectors = 1
print('Querying Weaviate...')
result = query(input_text, k_vectors)
print(json.dumps(result, indent=4))
