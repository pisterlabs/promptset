# lib_vectordb.py
import os
import logging
from dotenv import load_dotenv

# Assuming some document database abstraction is available as model_abstraction
from .lib_model import get_vectordb
from langchain_core.documents import Document

# Initialize the document database (vectordb)
vectordb = get_vectordb()

def add_doc(doc: Document, obj_id: str):
    """
    Add a langchain Document object to the vectordb.

    Parameters:
    doc (Document): The langchain Document object to add.
    obj_id (str): The unique identifier for the document.
    """
    logger = lib_logging.get_logger()
    try:
        text = doc.page_content
        metadata = doc.metadata
        # Using the provided method to add a single document
        vectordb.add_texts([text], metadatas=[metadata])
        logger.info(f"Successfully added document: {obj_id}")
    except Exception as e:
        logger.error(f"Error adding document {obj_id}: {e}")

def bulk_add_docs(docs: list):
    """
    Add multiple langchain Document objects to the vectordb in bulk.

    Parameters:
    docs (list of Document): A list of langchain Document objects to add.
    """
    logger = lib_logging.get_logger()
    try:
        texts = [doc.page_content for doc in docs]
        metadatas = [doc.metadata for doc in docs]
        # Using the provided method to add multiple documents in bulk
        vectordb.add_texts(texts, metadatas=metadatas)
        logger.info(f"Successfully added {len(docs)} documents in bulk")
    except Exception as e:
        logger.error(f"Error adding documents in bulk: {e}")