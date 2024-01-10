from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.schema import Document
from typing import List
import streamlit as st

def process_db_data() -> List[Document]:
    chunk_size = st.session_state["chunk_size"]
    chunk_overlap = st.session_state["chunk_overlap"]
    print(f"Chunk Size {chunk_size} Overlap {chunk_overlap}")
    loader = TextLoader("output.txt")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)