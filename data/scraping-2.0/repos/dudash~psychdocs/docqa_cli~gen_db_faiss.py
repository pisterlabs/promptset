# Originally from: https://gist.github.com/kennethleungty/7865e0c66c79cc52e9db9aa87dba3d59#file-db_build-py
# Original author: Kenneth Leung
# Snapshot date: 2023-08-01
#
# What's happening here: 
# - Data ingestion and splitting text into chunks
# - Load embeddings model (sentence-transformers)
# - Index chunks and store in FAISS vector store

import timeit
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings

print("Beginning FAISS vectorstore generation, this may take several minutes...")
start = timeit.default_timer()

# Load alls PDF files from data path using langchain's DirectoryLoader
print(f"Loading PDFs from data path...")
loader = DirectoryLoader('data/', glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True, use_multithreading=True)
documents = loader.load()
# print the number of documents loaded
print(f"Loaded directory in a list of {len(documents)} document pages")

# Split text from PDF into chunks - we have to do this because to not overload the embeddings model token size
print(f"Splitting documents into chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(documents)
print(f"Split document pages into {len(texts)} chunks")

# Load embeddings model
# https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                model_kwargs={'device': 'cpu'})

# Build and persist FAISS vector store
print(f"Building FAISS vectorstore from documents...")
vectorstore = FAISS.from_documents(texts, embeddings)
vectorstore.save_local('vectorstore/db_faiss')

end = timeit.default_timer()

print(f"Time to generate FAISS vectorstore: {end - start} seconds (or {(end - start)/60} minutes)")