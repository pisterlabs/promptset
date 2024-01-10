import os
from azure_file_loader import azureLoader
from embedding import get_embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, Pinecone
import pinecone
from text_splitter import get_text_splitter

# PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
# PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_API_ENV = os.environ["PINECONE_API_ENV"]
# index_name = "langchain2"


# create new index in pinecone with list in strings after splitting
def create_new_index(index_name):
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)
    embeddings = get_embeddings()
    texts = get_text_splitter()
    Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)
    
    print(f"Index: {index_name} created")

    return f"Index: {index_name} created in pinecone"

# load the pinecone instance with the index name and embeddings 
def load_pinecone(index_name):
    embeddings = get_embeddings()
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)
    return Pinecone.from_existing_index(index_name, embeddings)
    
