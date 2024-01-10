import glob
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY") or "OPENAI_API_KEY")
import pinecone
import time
import re

from uuid import uuid4

from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv(), override=True)  # read local .env file

pinecone_api_key = os.getenv("PINECONE_API_KEY") or "YOUR_API_KEY"
pinecone_env = os.getenv("PINECONE_ENVIRONMENT") or "YOUR_ENV"

base_dir = "./examples/src/examples"

loader = GenericLoader.from_filesystem(
    base_dir,
    glob="**/*",
    suffixes=[".ts"],
    parser=LanguageParser(),
)

docs = loader.load()

model_name = "text-embedding-ada-002"

texts = [c.page_content for c in docs]


def extract_comments_from_ts_code(ts_code):
    comment_pattern = r"(\/\/[^\n]*|\/\*[\s\S]*?\*\/)"
    comments = re.findall(comment_pattern, ts_code)
    comments_string = " ".join(
        comment.strip("/*").strip("*/").strip("//").strip() for comment in comments
    )

    return comments_string


metadatas = [
    "Filename: "
    + c.metadata["source"]
    + " Code content: "
    + extract_comments_from_ts_code(c.page_content)
    for c in docs
]

chunks = [
    texts[i : (i + 1000) if (i + 1000) < len(texts) else len(texts)]
    for i in range(0, len(texts), 1000)
]
embeds = []

print("Have", len(chunks), "chunks")
print("Last chunk has", len(chunks[-1]), "texts")

for chunk, i in zip(chunks, range(len(chunks))):
    print("Chunk", i, "of", len(chunk))
    new_embeddings = client.embeddings.create(input=chunk, model=model_name)
    new_embeds = [emb.embedding for emb in new_embeddings.data]

    embeds.extend(new_embeds)

pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)

index_name = "zkappumstad"

""" if index_name in pinecone.list_indexes():
    pinecone.delete_index(index_name)

pinecone.create_index(
    name=index_name,
    metric='dotproduct',
    dimension=1536
)  """

time.sleep(5)

while not pinecone.describe_index(index_name).status["ready"]:
    time.sleep(1)

index = pinecone.Index(index_name)

ids = [str(uuid4()) for _ in range(len(docs))]

vector_type = os.getenv("CODE_VECTOR_TYPE") or "CODE_VECTOR_TYPE"

vectors = [
    (
        ids[i],
        embeds[i],
        {
            "text": texts[i],
            "title": metadatas[i],
            "vector_type": vector_type,
        },
    )
    for i in range(len(docs))
]

for i in range(0, len(vectors), 100):
    batch = vectors[i : i + 100]
    print("Upserting batch:", i)
    index.upsert(batch)

print(index.describe_index_stats())
print("Example loader completed!")
