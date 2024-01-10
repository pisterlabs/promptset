import glob
import os
import openai
import pinecone
import time
import re
import json

from uuid import uuid4

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv(), override=True)  # read local .env file

pinecone_api_key = os.getenv("PINECONE_API_KEY") or "YOUR_API_KEY"
pinecone_env = os.getenv("PINECONE_ENVIRONMENT") or "YOUR_ENV"

pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY") or "OPENAI_API_KEY")

index_name = "zkappumstad"
model_name = "text-embedding-ada-002"
dep_path = './deprecated_code.json'

texts = []
metadatas = []

try:
    with open(dep_path, "r") as file:
        dep_data = file.read()

    dep = json.loads(dep_data)

    for code in dep:
        comment = code['comment']
        code = code['code']

        element = comment + "Code: " + code
        texts.append(element)
        metadatas.append(element)
except (KeyError, json.JSONDecodeError) as e:
    print(f"Error processing file {dep_path}: {e}")

chunks = [
    texts[i : (i + 1000) if (i + 1000) < len(texts) else len(texts)]
    for i in range(0, len(texts), 1000)
]
embeds = []

print("Have", len(chunks), "chunks")
print("Last chunk has", len(chunks[-1]), "texts")

for chunk, i in zip(chunks, range(len(chunks))):
    print("Chunk", i, "of", len(chunk))
    new_embeddings = client.embeddings.create(input=chunk, model=model_name,)

    new_embeds = [emb.embedding for emb in new_embeddings.data]
    embeds.extend(new_embeds)
    print(len(embeds))
    #  add time sleep if you encounter embedding token rate limit issue
    time.sleep(2)

while not pinecone.describe_index(index_name).status["ready"]:
    time.sleep(1)

index = pinecone.Index(index_name)

ids = [str(uuid4()) for _ in range(len(texts))]

vector_type = os.getenv("ISSUE_VECTOR_TYPE") or "ISSUE_VECTOR_TYPE"

vectors = [
    (
        ids[i],
        embeds[i],
        {"text": texts[i], "title": metadatas[i], "vector_type": vector_type},
    )
    for i in range(len(texts))
]

for i in range(0, len(vectors), 100):
    batch = vectors[i : i + 100]
    print("Upserting batch:", i)
    index.upsert(batch)

print(index.describe_index_stats())
print("Deprecated Loader Completed!")
