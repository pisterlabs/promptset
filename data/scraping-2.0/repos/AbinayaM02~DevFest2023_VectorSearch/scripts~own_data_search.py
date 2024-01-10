""" Script to run search on your own document.
"""

# Import necessary libraries
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
import os
from config import FAISS_PATH


def get_document(uploaded_file: object) -> list:
    """ Get the document and extract the pages.

    Args:
        uploaded_file (object): data file.

    Returns:
        list: list of pages.
    """
    # Load document if file is uploaded
    loader = PyPDFLoader(uploaded_file)
    pages = loader.load_and_split()
    return pages

def get_embeddings_model(emb_model: str) -> object:
    """ Get the embedding model.

    Args:
        emb_model (str): model name.

    Returns:
        object: embedding model object.
    """
    # Select embeddings
    embeddings = HuggingFaceEmbeddings(model_name = emb_model)
    return embeddings

def index_embeddings(texts: list, 
                     embeddings: object,
                     tmp_file: str) -> object:
    """ Obtain the embeddings and index it locally.

    Args:
        texts (list): list of extracted pages.
        embeddings (object): embedding model.
        tmp_file (str): uploaded file name.

    Returns:
        object: faiss index object.
    """
    # Create a vectorstore from documents if it doesn't exist
    if not os.path.exists(FAISS_PATH + tmp_file):   
        db = FAISS.from_documents(texts, embeddings)
        db.save_local(FAISS_PATH + tmp_file)
    # Load existing embedding for the same file
    else:
        db = FAISS.load_local(FAISS_PATH + tmp_file, embeddings)
    return db

def get_response(db: object, query: str) -> list:
    """ Get the matching results.

    Args:
        db (object): faiss index object.
        query (str): user query.

    Returns:
        list: list of matching pages.
    """
    response = db.similarity_search(query)
    return response

    
