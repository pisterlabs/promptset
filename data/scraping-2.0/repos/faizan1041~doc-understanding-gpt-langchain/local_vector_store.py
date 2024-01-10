from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import ReadTheDocsLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import UnstructuredWordDocumentLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from config.config import *
import hashlib
import os

def get_unique_index_name(path, index_type):
    pdf_hash = hashlib.md5(path.encode()).hexdigest()  # Calculate MD5 hash of the PDF path
    unique_index_name = f'faiss_index_{index_type}_{pdf_hash}'
    return unique_index_name

def ingest_docs(path, index_type):
    unique_index_name = get_unique_index_name(path, index_type)
    index_path = os.path.join(OUTPUT_DIR, unique_index_name)

    if os.path.exists(index_path):
        embeddings = OpenAIEmbeddings()
        new_vectorstore = FAISS.load_local(index_path, embeddings)
        return new_vectorstore
    
    if path.endswith(".pdf"):
        loader = PyPDFLoader(path)
        document = loader.load()
        print(f"loaded {len(document)} documents")

    elif path.endswith(".docx"):
        # loader = UnstructuredWordDocumentLoader(path)
        loader = Docx2txtLoader(path)
        document = loader.load()
        print(f"loaded {len(document)} documents")

    else:
        loader = ReadTheDocsLoader(path, encoding="utf-8")
        document = loader.load()
        print(f"loaded {len(document)} documents")
    

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=100, separators=["\n\n", "\n", " ", ""]
    )
    documents = text_splitter.split_documents(documents=document)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(index_path)

    new_vectorstore = FAISS.load_local(index_path, embeddings)
    print("****** Added to FAISS vectorstore vectors")
    return new_vectorstore



