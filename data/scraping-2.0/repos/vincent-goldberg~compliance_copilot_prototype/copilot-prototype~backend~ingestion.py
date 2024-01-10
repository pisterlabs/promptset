# Ingest packages
import os
import torch
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain import HuggingFaceHub

# Global variables
load_dotenv()   # Load environment variables from .env file
huggingfacehub_api_token = os.getenv("HUGGINGFACE_API_KEY")
mistral_repo = 'mistralai/Mistral-7B-Instruct-v0.1'


# Tokenizer
embedd_model = 'BAAI/bge-reranker-large'
model_kwargs = {"device": 'cpu'}
encode_kwargs = {"normalize_embeddings": True}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=embedd_model, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

def ingest_doc(doc_path, file_name):
    
    # Checking if vector database exists, creating it if not
    outdir = "./backend/vector_databases/"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    # Creating database path
    db_path = os.path.join(outdir, file_name)
    print('Db Path: ', db_path)

    # Checking if the database already exists, and creating it if it doesn't
    if not os.path.exists(db_path):        
        # Loading doc
        loader = PyPDFLoader(doc_path)
        raw_doc = loader.load()

        # Split and store vectors
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                    chunk_overlap=20,
                                                    separators=["\n\n", "\n", " ", ""])
        all_splits = text_splitter.split_documents(raw_doc)


        # Creating vector store
        vectorstore = Chroma.from_documents(documents=all_splits, embedding=embeddings, persist_directory=db_path)    
    else:
        vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)                       

    return vectorstore
