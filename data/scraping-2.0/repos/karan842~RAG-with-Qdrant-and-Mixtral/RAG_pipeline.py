from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
import qdrant_client
from dotenv import load_dotenv
import os
load_dotenv()
qdrant_uri = os.getenv('QDRANT_URI')
qdrant_api_key = os.getenv('QDRANT_API_KEY')

'''
Building RAG pipeline using QdrantDB and LangChain  
'''

# Create a Qdrant Client
client = qdrant_client.QdrantClient(
    qdrant_uri,
    api_key=qdrant_api_key
)

# Create a collection
vectors_config = qdrant_client.http.models.VectorParams(
    size=384,
    distance=qdrant_client.http.models.Distance.COSINE
)

client.recreate_collection(
    collection_name="my-collection",
    vectors_config=vectors_config
)

# Define Embeddings using HF
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load document from data directory in .pdf format
def load_documents():
    loader = DirectoryLoader('data/', glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

# Split texts
def get_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(text)
    return chunks

documents = load_documents()
text_chunks = get_chunks(documents)

qdrant = Qdrant.from_documents(
    text_chunks,
    embeddings,
    qdrant_uri,
    qdrant_api_key,
    prefer_grpc=True,
    collection_name='my-collection',
)


