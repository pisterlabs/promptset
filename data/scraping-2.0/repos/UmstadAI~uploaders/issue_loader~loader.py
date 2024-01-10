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
base_dir = "./issues_json"

json_files = glob.glob(os.path.join(base_dir, "**/*.json"), recursive=True)

issues = []

for issue_path in json_files:
    try:
        with open(issue_path, "r") as file:
            issue_data = file.read()

        issue = json.loads(issue_data)

        if not issue.get("question") or not issue.get("answer"):
            continue

        if not issue.get("full_question"):
            question = issue["question"]
        else:
            question = issue["full_question"] + issue["question"]

        answer = issue["answer"]

        issues.append({"question": question, "answer": answer})

    except (KeyError, json.JSONDecodeError) as e:
        print(f"Error processing file {issue_path}: {e}")
        continue

texts = []
metadatas = []

for issue in issues:
    texts.append(
        "Question: " + str(issue["question"]) + "\n Answer: " + str(issue["answer"])
    )
    metadatas.append(issue["question"])

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

ids = [str(uuid4()) for _ in range(len(issues))]

vector_type = os.getenv("ISSUE_VECTOR_TYPE") or "ISSUE_VECTOR_TYPE"

vectors = [
    (
        ids[i],
        embeds[i],
        {"text": texts[i], "title": metadatas[i], "vector_type": vector_type},
    )
    for i in range(len(issues))
]

for i in range(0, len(vectors), 100):
    batch = vectors[i : i + 100]
    print("Upserting batch:", i)
    index.upsert(batch)

print(index.describe_index_stats())
print("Issue Loader Completed!")