import streamlit as st
import os
from langchain.document_loaders import (
    CSVLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from configs.apikey import apikey

os.environ["OPENAI_API_KEY"] = apikey  

LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".pdf": (PyMuPDFLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
}


def getLoader(pdf_path, ext):
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(pdf_path, **loader_args)
        return loader.load()


def injest(pdf_path, ext, chunk_size):
    persist_directory = "db/chroma"
    documents = getLoader(pdf_path, ext)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name="fusion-ai",
    )
    vectordb.persist()

def create_vector_store(uploaded_file, chunk_size):
    st.spinner(text="In progress...")
    print("data/" + uploaded_file.name)
    file_extension = uploaded_file.name.split(".")[-1]
    injest("data/" + uploaded_file.name, "."+file_extension, chunk_size)
    st.success("Vector Store is Created Successfully!")