from decouple import config
import pandas as pd
import numpy as np
import pinecone # for vector database
import openai
from openai.embeddings_utils import distances_from_embeddings, cosine_similarity

openai.api_key = str(config('API_OPENAI'))
index_name = 'football-buzz'
dimensions = 1536

pinecone.init(
    api_key=str(config('API_PINECONE')),
    environment="us-east-1-aws")

#if index_name in pinecone.list_indexes():
    # pinecone.delete_index(index_name)
#pinecone.create_index(name=index_name, dimension=dimensions, metric="cosine")

# Initialize the Pinecone index instance
index = pinecone.Index(index_name=index_name)

df=pd.read_csv('processed/embed/embed-comb.csv', index_col=0)
df.columns = ['batchid', 'text', 'tokens', 'embeddings']
df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)
print(df.head())

vectors = [(str(row['batchid']), row['embeddings'].tolist(), {'text': row['text']}) for _, row in df.iterrows()]

batch_size = 100
num_batches = len(vectors) // batch_size + 1

for i in range(num_batches):
    batch_start = i * batch_size
    batch_end = (i + 1) * batch_size
    batch_vectors = vectors[batch_start:batch_end]
    
    upsert_response = index.upsert(
        vectors=batch_vectors,
        namespace=index_name, values=True, include_metadata=True)
    
    print(f"Upserted {len(batch_vectors)} vectors from index {batch_start} to {batch_end}")

query = "Is Aaron Rodgers the best QB of all time?"
response = openai.Embedding.create(input=query, model='text-embedding-ada-002')
query_response_embeddings = response['data'][0]['embedding']

vector_database_results_matching = index.query([query_response_embeddings], top_k=5, include_metadata=True, include_Values=True, 
    namespace=index_name)
for match in vector_database_results_matching['matches']:
    print(f"{match['score']:.2f}: ")
    print(f"{match['score']:.2f}: {match['metadata']['text']}")