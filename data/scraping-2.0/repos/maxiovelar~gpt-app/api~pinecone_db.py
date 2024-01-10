import os
import pinecone
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings



################################################################################
# CONFIG
################################################################################
INDEX_NAME = "langchain-products"


################################################################################
# Initialize Pinecone instance
################################################################################
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENV"),
)


def get_embeddings():
    return OpenAIEmbeddings(
        openai_api_key = os.getenv('OPENAI_API_KEY')
    )

################################################################################
# Get database instance
################################################################################
def get_db_instance():
    embeddings = get_embeddings()
    return Pinecone.from_existing_index(INDEX_NAME, embeddings)


################################################################################
# Generate embeddings on pinecone database
################################################################################
def get_db_embeddings(docs):
    embeddings = get_embeddings()
    
    if INDEX_NAME not in pinecone.list_indexes():
        pinecone.create_index(
        name = INDEX_NAME,
        metric = 'cosine',
        dimension = 1536
    )

    return Pinecone.from_documents(
        docs,
        embeddings,
        index_name = INDEX_NAME
    )


def delete_index():
    pinecone.delete_index(INDEX_NAME)
    