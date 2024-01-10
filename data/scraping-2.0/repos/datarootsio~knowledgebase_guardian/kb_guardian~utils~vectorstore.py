import pickle
from typing import List

import faiss
from langchain.schema import Document
from langchain.vectorstores import FAISS

from kb_guardian.utils.deployment import get_deployment_embedding


def create_FAISS_vectorstore(document_chunks: List[Document]) -> FAISS:
    """
    Create a FAISS vector store from a list of LangChain Document chunks.

    Args:
        document_chunks (List[Document]): A list of LangChain Document chunks
          to create the vector store from.

    Returns:
        FAISS: The created FAISS vector store.

    """
    embedding = get_deployment_embedding()
    vectorstore = FAISS.from_documents(document_chunks, embedding)
    return vectorstore


def load_FAISS_vectorstore(index_filename: str, store_filename: str) -> FAISS:
    """
    Load a FAISS vector store from the given index and store files.

    Args:
        index_filename (str): The filename of the FAISS index file.
        store_filename (str): The filename of the vector store file.

    Returns:
        FAISS: The loaded FAISS vector store.

    """
    index = faiss.read_index(index_filename)
    with open(store_filename, "rb") as f:
        vectorstore = pickle.load(f)
    vectorstore.index = index
    return vectorstore


def save_FAISS_vectorstore(
    vectorstore: FAISS, index_filename: str, store_filename: str
) -> None:
    """
    Save a FAISS vector store to the given index and store files.

    Args:
        vectorstore (Any): The FAISS vector store to be saved.
        index_filename (str): The filename of the FAISS index file.
        store_filename (str): The filename of the vector store file.

    Returns:
        None

    """
    faiss.write_index(vectorstore.index, index_filename)
    vectorstore.index = None
    with open(store_filename, "wb") as f:
        pickle.dump(vectorstore, f)
