import os
import time
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from dotenv import load_dotenv

load_dotenv()

# find API key in console at app.pinecone.io
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") or "PINECONE_API_KEY"
# find ENV (cloud region) next to API key in console
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT") or "PINECONE_ENVIRONMENT"


def create_store(index_name, documents, openai_api_key):
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002", openai_api_key=openai_api_key
    )

    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENVIRONMENT,
    )

    # First, check if our index already exists. If it doesn't, we create it
    if index_name not in pinecone.list_indexes():
        # we create a new index
        # The OpenAI embedding model `text-embedding-ada-002 uses 1536 dimensions`
        print(f"Creating index {index_name}...")
        pinecone.create_index(name=index_name, metric="cosine", dimension=1536)
    else:
        print(f"Index {index_name} already exists.")

    docsearch = Pinecone.from_documents(documents, embeddings, index_name)

    print("Connecting to index...")
    index = pinecone.Index(index_name)
    # wait a moment for the index to be fully initialized
    time.sleep(1)
    print("Index info:")
    index.describe_index_stats()

    print("Done.")

    return docsearch


def get_store(index_name, openai_api_key):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # Create a Pinecone client using the existing index and SentenceTransformer embeddings
    docsearch = Pinecone.from_existing_index(index_name, embeddings)

    if docsearch is None:
        raise ValueError("No db found for the given repo names.")

    return docsearch
