import os
from dotenv import load_dotenv
import pinecone
import pandas as pd
from openai import OpenAI

load_dotenv()

client = OpenAI(
  api_key=os.environ['OPENAI_API_KEY']
)

df = pd.read_json('outputs/combined_data_first_200_rows.jsonl', lines=True)

df['combined_text'] = df.apply(lambda x: f"{x['product']} {x['summary']} {' '.join(x['categories'])}", axis=1)

pinecone.init(api_key=os.environ.get("PINECONE_API_KEY"), environment='us-west4-gcp-free')

EMBEDDING_DIMENSION = 2048

index_name = 'green'
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=EMBEDDING_DIMENSION)
index = pinecone.Index(index_name)

print("=" * 10 + " Pinecone Index Initialized " + "=" * 10)

def get_embeddings(texts):
    response = client.embeddings.create(input=texts, model="text-similarity-babbage-001")
    return [embedding.embedding for embedding in response.data]

embeddings = get_embeddings(df['combined_text'].tolist())

print("=" * 10 + " OpenAI Embeddings Created " + "=" * 10)

to_upload = [(str(id), embedding) for id, embedding in zip(df['id'], embeddings)]
index.upsert(vectors=to_upload)

print("=" * 10 + " OpenAI Embeddings Uploaded to Pinecone " + "=" * 10)