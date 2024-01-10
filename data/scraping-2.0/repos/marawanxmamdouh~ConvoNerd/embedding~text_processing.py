# Importing the necessary libraries
from typing import List

from box import Box
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from utils.helpers import get_config

# Get the configuration
cfg: Box = get_config('text_processing.yaml')


def get_text_chunks(data: str | list[Document]) -> list[str] | list[Document]:
    """
    Splits the input data into chunks using a text splitter.

    Parameters
    ----------
    data: str or list[Document]
        A string containing the document to be split,
        or a list of documents where each document is represented as a list of sentences.

    Returns
    -------
    chunks: list[str] | list[Document]
        A list of chunks from the original text.

    """
    # Initialize the text splitter
    text_splitter: RecursiveCharacterTextSplitter = initialize_text_splitter()

    # Check if the text is a string
    if isinstance(data, str):
        chunks: list[str] = split_text(data, text_splitter)
    else:
        chunks: list[Document] = split_documents(data, text_splitter)

    return chunks


def initialize_text_splitter() -> RecursiveCharacterTextSplitter:
    """
    Initializes the text splitter using specific parameters.

    Returns
    -------
    RecursiveCharacterTextSplitter
        An instance of RecursiveCharacterTextSplitter initialized with specific parameters.

    """
    return RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        length_function=len
    )


def split_text(text: str, text_splitter: RecursiveCharacterTextSplitter) -> list[str]:
    """
    Splits a document into chunks.

    Parameters
    ----------
    text: str
        A string containing the text data to be split.
    text_splitter: RecursiveCharacterTextSplitter
        An instance of RecursiveCharacterTextSplitter.

    Returns
    -------
    chunks : List[str]
        A list of chunks from the original text.
    """
    return text_splitter.split_text(text)


def split_documents(documents: list[Document], text_splitter: RecursiveCharacterTextSplitter) -> list[Document]:
    """
    Splits a list of documents into chunks.

    Parameters
    ----------
    documents: list[Document]
        A list of documents where each document is represented as a list of sentences.
    text_splitter: RecursiveCharacterTextSplitter
        An instance of RecursiveCharacterTextSplitter.

    Returns
    -------
    chunks: list[Document]
        A list of chunks from the original documents.

    """
    return text_splitter.split_documents(documents)
