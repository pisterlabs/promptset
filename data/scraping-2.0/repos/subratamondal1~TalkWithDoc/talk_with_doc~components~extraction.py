import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter

@st.cache_data
def extract_raw_text(docs:list):
    text = "" # Store the whole docs as a single string
    for doc in docs: # for handling multiple files
        pdf_reader = PdfReader(doc)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

@st.cache_data
def extract_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )

    text_chunks = text_splitter.split_text(raw_text)
    return text_chunks