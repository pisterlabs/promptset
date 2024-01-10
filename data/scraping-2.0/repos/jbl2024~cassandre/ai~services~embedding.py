"""
This module provides functions for working with language embeddings.
"""
from langchain.embeddings import OpenAIEmbeddings

# from langchain.embeddings import SentenceTransformerEmbeddings


def get_embedding():
    """
    Return an embedding model.

    Returns:
        OpenAIEmbeddings: An instance of the OpenAIEmbeddings class.
    """
    # return SentenceTransformerEmbeddings(model_name="hkunlp/instructor-xl")
    # return SentenceTransformerEmbeddings(model_name="BAAI/bge-small-en")
    return OpenAIEmbeddings()


# Represent this sentence for searching relevant passages:


def get_query_prefix():
    """
    Returns the query prefix for searching relevant passages.

    Returns:
        str: A string representing the query prefix.
    """
    return ""
    # return "Represent this sentence for searching relevant passages:"
