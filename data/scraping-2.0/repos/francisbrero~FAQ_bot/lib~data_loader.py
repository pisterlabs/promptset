"""
This module contains the data loader for loading the help articles.
"""
import os
import json

import requests

from bs4 import BeautifulSoup
from langchain.schema.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import Chroma
from chroma import Chroma

VECTOR_DB_FOLDER = "data/kb_vector_db"


def extract_documents(url: str):
    """
    Extracts documents from a given URL.

    Args:
        url (str): The URL to extract documents from.

    Returns:
        List[Document]: A list of documents extracted from the URL.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")
        paragraphs = soup.find_all(class_="accordion__item-content")

        return [
            Document(
                page_content=paragraph.get_text(),
                metadata={"source": url},
            )
            for paragraph in paragraphs
        ]
    except requests.exceptions.RequestException as e:
        print("Error occurred while fetching the URL:", e)
        return []


def build_vector_db(embeddings):
    """
    Builds the vector database from the help articles.

    Args:
        embeddings: The embeddings to use for the vector database.

    Returns:
        The vector database.
    """
    urls = [data["url"] for data in json.load(open("../data/scraped_urls_website.json"))]

    all_docs = []
    for url in urls:
        docs = extract_documents(url)
        # TODO there are some URLs with no documents extracted, need to update extract_documents
        with open("../data/logs.txt", "a") as f:
            f.write(f"Extracted {len(docs)} documents from {url}\n")
        all_docs.extend(docs)

    vector_db = Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        persist_directory=VECTOR_DB_FOLDER,
        collection_metadata={"hnsw:space": "cosine"},
    )
    vector_db.persist()

    return vector_db


def get_vector_db():
    """
    Gets the vector database.

    Returns:
        The vector database.
    """
    embeddings = HuggingFaceEmbeddings()
    if not os.path.exists(VECTOR_DB_FOLDER):
        return build_vector_db(embeddings)
    else:
        return Chroma(persist_directory=VECTOR_DB_FOLDER, embedding_function=embeddings)
