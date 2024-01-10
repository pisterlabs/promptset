import os
import tempfile

import streamlit as st

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.embeddings import OpenAIEmbeddings

from utils import file_formats_dataframe


@st.cache_resource(ttl="1h")
def configure_retriever(files, api_key):
    docs = []
    temp_dir = tempfile.TemporaryDirectory()

    for file in files:
        print(f"Procesando archivo {file.name}...")

        temp_filepath = os.path.join(temp_dir.name, file.name)

        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())

        if file.name.endswith(".docx"):
            print("Procesando archivo .docx...")
            loader = Docx2txtLoader(temp_filepath)
            docs.extend(loader.load())

        elif file.name.endswith(".pdf"):
            print("Procesando archivo .pdf...")
            loader = PyPDFLoader(temp_filepath)
            docs.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    print("Creando embeddings...")

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectordb = FAISS.from_documents(splits, embeddings)

    return vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 4})


@st.cache_data(ttl="1h")
def load_data(uploaded_file):
    try:
        ext = os.path.splitext(uploaded_file.name)[1][1:].lower()
    except Exception as e:  # noqa
        ext = uploaded_file.split(".")[-1]
    if ext in file_formats_dataframe:
        return file_formats_dataframe[ext](uploaded_file)
    else:
        st.error(f"Formato incorrecto: {ext}")
        return None
