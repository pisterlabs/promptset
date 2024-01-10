import logging

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_documents(document_paths: list[str]) -> list[str]:
    """
    Loads documents from the given paths using PyPDFLoader.

    Args:
        document_paths (list[str]): List of paths to PDF documents.

    Returns:
        list[str]: List of loaded documents.
    """
    docs = []  # Initialize an empty list to store the contents of the documents.

    try:
        # Create a list of PyPDFLoader objects, each initialized with a path from document_paths.
        loaders = [PyPDFLoader(path) for path in document_paths]

        # Iterate through each loader and load the document content.
        # PyPDFLoader.load() reads the content of the PDF file located at the given path.
        for loader in loaders:
            docs.extend(loader.load())  # Extend the docs list with the content of each loaded document.

    except Exception as e:
        # Log any exceptions encountered during the document loading process.
        # This could include file not found errors, issues with reading the PDF, etc.
        logging.error(f'Error loading documents: {e}')

    return docs  # Return the list of loaded document contents.


def split_documents(docs: list[str], chunk_size: int = 200, chunk_overlap: int = 50) -> list[str]:
    """
    Splits documents into smaller chunks.

    Args:
        docs (list[str]): List of documents to be split.
        chunk_size (int): Size of each chunk in characters.
        chunk_overlap (int): Overlap between chunks in characters.

    Returns:
        list[str]: List of split document chunks.
    """
    try:
        # Initialize text splitter and split each document into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return text_splitter.split_documents(docs)
    except Exception as e:
        # Log any exceptions encountered during document splitting
        logging.error(f'Error splitting documents: {e}')
        return []