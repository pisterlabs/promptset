import streamlit as st
from unstructured.partition.auto import partition
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()


def extract_file(file):
    """
    Process extraction data in streamlit server.
    """
    try:
        elements = partition(file)
        return "\n\n".join([str(el) for el in elements])
    except Exception as e:
        st.error(f"Failed to extract text: {e}")
        return None


CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100


def get_document_embeddings(text):
    # Split all text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE,
                                                   chunk_overlap=CHUNK_OVERLAP)
    splits = text_splitter.split_text(text)
    # Create a Vector Store
    docembeddings = FAISS.from_texts(splits, OpenAIEmbeddings())
    docembeddings.save_local("documents_faiss_index")
    docembeddings = FAISS.load_local("documents_faiss_index",
                                     OpenAIEmbeddings())
    return docembeddings
