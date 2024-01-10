import logging
import os

import pinecone

# load environment variable
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone

load_dotenv()

# Initialize all required env variables

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or "OPENAI_API_KEY"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") or "PINECONE_API_KEY"
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT") or "PINECONE_ENVIRONMENT"
# PINECONE index name (knowledgebase)
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME") or "PINECONE_INDEX_NAME"


def initialize_vector_store(
    index_name=PINECONE_INDEX_NAME, model_name="text-embedding-ada-002"
):
    # model_name = 'text-embedding-ada-002'
    # index_name = 'sutton-barto-retrieval-augmentation'
    text_field = "text"
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

    # Initialize pinecone index
    index = pinecone.Index(index_name)

    # embedding model for input
    embed = OpenAIEmbeddings(model=model_name, openai_api_key=OPENAI_API_KEY)
    # Initialize vector store for querying
    vectorstore = Pinecone(index, embed.embed_query, text_field)
    logging.info(f"Index Found. Stats: {index.describe_index_stats()}")

    return [vectorstore, index]
