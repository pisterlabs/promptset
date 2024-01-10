"""
This file extracts the PID, property label, and property description from Wikidata.
It then uses the OpenAI embeddings to embed the property label and description.
The embeddings are then pushed to the Pinecone index to use in nearest neighbor search in our algorithm.
"""

# Use pinecone to push the PIDs to the index
import json
import pinecone
from langchain.embeddings import OpenAIEmbeddings
from tqdm import tqdm

# Load the PID dictionary
with open("pid_to_label.json", "r") as f:
    pid_to_info = json.load(f)

# Connect to the index
embeddings_model = OpenAIEmbeddings(openai_api_key="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")
pinecone_model = pinecone.init(api_key="53ad2ca2-3d03-45a3-b4c3-36baa6ca835a", environment="gcp-starter")
pinecone_index = pinecone.Index('kraft')

embeddings = []
chunks = []
pids = []

# Iterate through the PID dictionary and push the embeddings to the index
for pid, (label, description, aliases) in tqdm(pid_to_info.items()):
    chunk = f'{label} {description}'
    chunks.append(chunk)
    pids.append(pid)

    # Push the embeddings to the index in chunks of 100, to not overload the API
    if len(chunks) == 100:
        embeddings = embeddings_model.embed_documents(chunks)
        pinecone_index.upsert(vectors=zip(pids, embeddings, [{'pid': pid} for pid in pids]))
        chunks = []
        pids = []

# Push the remaining embeddings to the index
embeddings = embeddings_model.embed_documents(chunks)
pinecone_index.upsert(vectors=zip(pids, embeddings, [{'pid': pid} for pid in pids]))
