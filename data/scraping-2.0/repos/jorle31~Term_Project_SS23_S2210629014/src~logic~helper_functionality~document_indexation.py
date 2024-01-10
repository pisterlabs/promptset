"""
File that contains the logic for indexation of documents.
"""
import re
import logging
from typing import List, Sequence, Literal, Optional

from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
import pinecone 

from src.logic.config import secrets as config_secrets

class Indexer():
    """
    Class that contains the logic/tools for the risk types index.
    """

    def __init__(self) -> None:
        pinecone.init(
            api_key = config_secrets.read_pinecone_credentials(),
            environment = "us-west4-gcp"
        )

    def do_indexation(self, documents: List[Document], namespace: Optional[str] = None, index_name: Optional[str] = "index", metric: Optional[str] = "cosine", pod_type: Optional[str] = "p1.x1") -> None:
        """
        Index a document in a vectorstore. Creates a new index if an index with the provided name 
        doesn't already exist.

        :param documents: The documents to be indexed.
        :param namespace: The namespace to be used for the index.
        :param index_name: The name of the index to be used.
        :param metric: The metric to be used for the index.
        :param pod_type: The pod type to be used for the index.
        :return: None
        :raise ValueError: If arg documents is not a list of Documents or if the list is empty.
        :raise TypeError: If arg namespace is not a string.
        :raise ValueError: If arg index_name is not a string or if it doesn't match the pattern.
        :raise TypeError: If arg metric is not a string.
        :raise TypeError: If arg pod_type is not a string.
        """
        if not isinstance(documents, Sequence) or not all(isinstance(doc, Document) for doc in documents):
            raise ValueError("Argument documents must be a non-empty list of Documents.")
        if not isinstance(namespace, str):
            raise TypeError("Argument namespace must be a string.")
        if not isinstance(index_name, str):
            raise ValueError("Argument index_name must be a string.")
        pattern: Literal = r"^[a-z0-9][a-z0-9-]*[a-z0-9]$"
        if not re.match(pattern, index_name):
            raise ValueError("""Argument index_name must consist of lower case alphanumeric characters or '-', and
            must start and end with an alphanumeric character.""")
        if not isinstance(metric, str):
            raise TypeError("Argument metric must be a string.")
        if not isinstance(pod_type, str):
            raise TypeError("Argument pod_type must be a string.")
        try:
            chunk_size: int = 1000
            chunk_overlap: int = 200
            split_documents: List[Document] = RecursiveCharacterTextSplitter(
                chunk_size = chunk_size, chunk_overlap = chunk_overlap
            ).split_documents(documents = documents)
            embeddings: OpenAIEmbeddings = OpenAIEmbeddings(
                openai_api_key = config_secrets.read_openai_credentials()
            )
            res: List[List[float]] = embeddings.embed_documents([doc.page_content for doc in split_documents])
            if index_name not in pinecone.list_indexes():
                pinecone.create_index(
                    name = index_name,
                    metric = metric,
                    dimension = len(res[0]),
                    pod_type = pod_type
                )
                Pinecone.from_documents(documents = split_documents, embedding = embeddings, index_name = index_name, namespace = namespace)
            else:
                index: Pinecone = Pinecone.from_existing_index(index_name = index_name, embedding = embeddings, namespace = namespace)
                index.add_documents(documents = split_documents, namespace = namespace)
        except (ValueError, TypeError) as e:
            logging.error(e)
            raise ValueError(f"Error: {e}") from e