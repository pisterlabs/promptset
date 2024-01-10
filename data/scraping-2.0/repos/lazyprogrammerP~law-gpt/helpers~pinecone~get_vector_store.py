import os

import pinecone
from langchain.vectorstores import Pinecone

from common.embeddings import embeddings

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

INDEX_NAME = "ai-ml-marvels-tam-vit-2023"


def get_vector_store():
    if INDEX_NAME not in pinecone.list_indexes():
        pinecone.create_index(name=INDEX_NAME, metric="cosine", dimension=768)

    return Pinecone.from_existing_index(index_name=INDEX_NAME, embedding=embeddings)
