#!/usr/bin/env python3

"""
This module provides functions for interacting with Pinecone indexes.

It includes functions for creating an index, populating it with embeddings,
fetching embeddings from an existing index, and deleting an index.

Note: This module requires the 'pinecone' and 'langchain' libraries to be installed.
"""

from typing import List, TypeVar
from langchain.vectorstores import Pinecone

T = TypeVar("T")


def create_index(index_name: str) -> None:
    """
    Creates a Pinecone index with the given name.

    Args:
        index_name (str): The name of the index to create.

    Raises:
        ValueError: If the index name is already taken.
        pinecone.exceptions.PineconeException: If there is an error creating the index.
    """
    import pinecone

    if index_name in pinecone.list_indexes():
        raise ValueError(f"Index {index_name} already exists.")
    try:
        print(f"Creating index {index_name}... ", end="")
        pinecone.create_index(name=index_name, metric="cosine", dimension=1536)
        print(f"Done")
    except pinecone.exceptions.PineconeException as e:
        raise pinecone.exceptions.PineconeException(f"Error creating index: {str(e)}")


def create_vector_store(index_name: str, chunks: List[T]) -> Pinecone:
    """
    Populates a Pinecone index with embeddings for the input documents (chunks).

    Args:
        index_name (str): The name of the index to populate.
        chunks (List[T]): A list of data chunks to index.

    Returns:
        Pinecone: A Pinecone vector store object containing the indexed data.

    Raises:
        ValueError: If the index name does not exist.
        pinecone.exceptions.PineconeException: If there is an error indexing the documents.
    """
    import pinecone
    from langchain.vectorstores import Pinecone
    from langchain.embeddings.openai import OpenAIEmbeddings
    from text_utils.text_utils import num_tokens_and_cost

    num_tokens, cost = num_tokens_and_cost(chunks)
    # Prompt user whether they want to continue, quit if they don't
    while True:
        user_input = input(
            f"Cost Estimate: ${cost:.4f} for {num_tokens} tokens\n"
            f"Would you like to continue? (y/n)\n"
        )
        if user_input.lower() == "y":
            break
        elif user_input.lower() == "n":
            print("Exiting...")
            exit()
        else:
            print("Invalid input. Please try again.")

    embeddings = OpenAIEmbeddings()
    if index_name not in pinecone.list_indexes():
        raise ValueError(f"Index {index_name} does not exist.")
    try:
        print(f"Indexing documents... ", end="")
        vector_store = Pinecone.from_documents(
            chunks, embeddings, index_name=index_name
        )
        print(f"Done")
    except pinecone.exceptions.PineconeException as e:
        raise pinecone.exceptions.PineconeException(
            f"Error indexing documents: {str(e)}"
        )
    return vector_store


def fetch_vector_store(index_name: str) -> Pinecone:
    """
    Retrieves embeddings from an already existing Pinecone index.

    Args:
        index_name (str): The name of the index to fetch embeddings from.

    Returns:
        Pinecone: A Pinecone vector store object containing the fetched embeddings.

    Raises:
        ValueError: If the index name does not exist.
        pinecone.exceptions.PineconeException: If there is an error fetching the embeddings.
    """
    import pinecone
    from langchain.vectorstores import Pinecone
    from langchain.embeddings.openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings()
    if index_name not in pinecone.list_indexes():
        raise ValueError(f"Index {index_name} does not exist.")
    try:
        print(f"Fetching {index_name}... ", end="")
        vector_store = Pinecone.from_existing_index(
            index_name=index_name, embedding=embeddings
        )
        print(f"Done")
        stats = vector_store.get_pinecone_index(
            index_name=index_name
        ).describe_index_stats()
        print(
            f"Index contains {stats['total_vector_count']} vectors for as many chunks of data"
        )
    except pinecone.exceptions.PineconeException as e:
        raise pinecone.exceptions.PineconeException(
            f"Error fetching embeddings: {str(e)}"
        )
    return vector_store


def delete_pinecone_index(index_name: str) -> None:
    """
    Deletes a Pinecone index with the given name or all indices if index_name is "all".

    Args:
        index_name (str): Name of the index to delete. If "all", all indices will be deleted.

    Returns:
        None
    """
    import pinecone

    if index_name == "all":
        indices = pinecone.list_indexes()
        print(f"Deleting all {len(indices)} indices... ", end="")
        for index in indices:
            pinecone.delete_index(index)
        print("Done")
    else:
        print(f"Deleting index {index_name}... ", end="")
        pinecone.delete_index(index_name)
        print("Done")
