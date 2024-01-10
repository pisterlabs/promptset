# import multiple file formats with langchain wrappers
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2textLoader
from langchain.document_loaders import TextLoader
# import for rapid UI development
import streamlit as st
import os

# implement load_docs function with st decorator
@st.cache_data()
def load_docs():
    documents = []
    #all input files in docs folder
    for file in os.listdir('docs'):
        if file.endswith('.pdf'):
            pdf_path = './docs/' + file
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
        elif file.endswith('.docx') or file.endswith('.doc'):
            doc_path = './docs/' + file
            loader = Docx2txtLoader(doc_path)
            documents.extend(loader.load())
        elif file.endswith('.txt'):
            text_path = './docs/' + file
            loader = TextLoader(text_path)
            documents.extend(loader.load())

    return documents