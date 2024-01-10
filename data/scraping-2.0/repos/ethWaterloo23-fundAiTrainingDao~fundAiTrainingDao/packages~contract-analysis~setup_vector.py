# This file indexes all the vulnerabilities and their descriptions
# into an embedding vector, and then adds the embedding vector to the
# database. The user can then query the database with a vulnerability
# with their solidity code, and the database will return the closest matching
# vulnerability.


"""
For all files inside the vulnerabilities folder, index the vulnerabilities.
"""

from functools import cache
import os
from utils import get_org_token

import requests
import chromadb


# Langchain doesn't have a SolidtyLoader, so we'll use the TextLoader
from langchain.document_loaders import (
    DirectoryLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain.schema import Document


def query_vector(text: str) -> list[float]:
    """
    Takes input a list of strings and returns a list of vectors.
    Uses the HuggingFace pipeline API.
    Returns a list of 384 vectors.
    """

    model_id = "sentence-transformers/all-MiniLM-L6-v2"
    hf_token = get_org_token()

    api_url = (
        f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
    )
    headers = {"Authorization": f"Bearer {hf_token}"}

    response = requests.post(
        api_url,
        headers=headers,
        json={"inputs": text, "options": {"wait_for_model": True}},
    )
    assert response.status_code == 200, "Request failed"
    return response.json()


def load_vulnerabilities() -> list[Document]:
    """
    Loads all the vulnerabilities from the vulnerabilities folder.
    """
    sol_loader = DirectoryLoader(
        "vulnerabilities", glob="**/*.sol", loader_cls=TextLoader
    )

    return sol_loader.load()


def load_markdowns() -> list[Document]:
    """
    Loads all the markdowns from the vulnerabilities folder.
    """
    md_loader = DirectoryLoader(
        "vulnerabilities", glob="**/*.md", loader_cls=UnstructuredMarkdownLoader
    )

    return md_loader.load()


def add_to_db():
    pass


def get_filename_from_path(file_path):
    """
    Extract the filename from a given path.
    E.g. /abc/def.txt -> def.txt
    """
    filename = os.path.basename(file_path)
    return filename


def create_database():
    """
    Loads all the vulnerabilities from the vulnerabilities folder,
    and then indexes them into an embedding vector.
    """
    # Check if the database already exists
    if os.path.exists("solidity_vulnerabilities.chromadb"):
        print("Database already exists? Skipping db creation...")
        return

    # Load the documents
    print("Loading documents...")
    vulnerabilities = load_vulnerabilities()
    markdowns = load_markdowns()

    # Get the embeddings
    print("Getting embeddings...")
    vulnerability_vectors = [query_vector(v.page_content) for v in vulnerabilities]
    markdown_vectors = [query_vector(m.page_content) for m in markdowns]

    # Add the embeddings to the database
    print("Creating database...")
    client = get_chromadb_client()
    collection = client.create_collection("solidity_vulnerabilities")

    # Add the vulnerabilities to the database
    print("Adding vulnerabilities...")
    collection.upsert(
        embeddings=vulnerability_vectors,
        metadatas=[doc.metadata for doc in vulnerabilities],
        documents=[doc.page_content for doc in vulnerabilities],
        ids=[doc.metadata["source"] for doc in vulnerabilities],
    )

    # Add the markdowns to the database
    print("Adding markdowns...")
    collection.upsert(
        embeddings=markdown_vectors,
        metadatas=[doc.metadata for doc in markdowns],
        documents=[doc.page_content for doc in markdowns],
        ids=[doc.metadata["source"] for doc in markdowns],
    )

    print("Done!")


@cache
def get_chromadb_client():
    client = chromadb.PersistentClient(
        path="solidity_vulnerabilities.chromadb",
    )
    return client


def query_database(text: str):
    """
    Does a test by querying the database.
    """
    print("Computing embedding...")
    embedding = query_vector(text)
    print("Querying database...")
    client = get_chromadb_client()
    collection = client.get_collection("solidity_vulnerabilities")
    results = collection.query(
        query_embeddings=embedding,
        n_results=3,
    )
    print("Query complete!")
    print(results)
    return {"results": results}


if __name__ == "__main__":
    create_database()
    sol_reentrancy_text = """
    pragma solidity ^0.4.15;

contract Reentrance {

    function withdrawBalance_fixed_2(){
        // send() and transfer() are safe against reentrancy
        // they do not transfer the remaining gas
        // and they give just enough gas to execute few instructions    
        // in the fallback function (no further call possible)
        msg.sender.transfer(userBalance[msg.sender]);
        userBalance[msg.sender] = 0;
    }   
   
}
    """

    response = query_database(sol_reentrancy_text)
    assert len(response["results"]) > 0
