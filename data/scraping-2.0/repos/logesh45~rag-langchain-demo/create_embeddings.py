import os
import sys

import pinecone
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone

load_dotenv()

# initialize pinecone
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENV"),
)


def create_embeddings(uploaded_files):
    # Read documents
    docs = []
    index_name = "preloaded-index"
    print("file", uploaded_files)
    for file in uploaded_files:
        loader = PyPDFLoader(file)
        docs.extend(loader.load())

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Create embeddings and store in vectordb
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # First, check if our index already exists. If it doesn't, we create it
    if index_name not in pinecone.list_indexes():
        # we create a new index
        pinecone.create_index(name=index_name, metric="cosine", dimension=384)
        # The OpenAI embedding model `text-embedding-ada-002 uses 1536 dimensions`
        docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)
    else:
        print("Index already exists.")


# Specify the directory containing the PDF files
files_directory = 'files'
file_paths = []

# Check if the directory exists
if os.path.exists(files_directory):
    # List all files in the directory
    for filename in os.listdir(files_directory):
        if filename.endswith(".pdf"):
            file_paths.append(os.path.join(files_directory, filename))
else:
    print(f"Directory '{files_directory}' not found.")
    sys.exit()

# Check if PDF files are found
if not file_paths:
    print("Please add PDF documents to the files folder to continue.")
    sys.exit()

create_embeddings(file_paths)
