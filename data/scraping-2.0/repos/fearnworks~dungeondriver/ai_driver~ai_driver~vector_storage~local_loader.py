from typing import List, Iterable
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from loguru import logger
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


def load(dir_path: str, glob_pattern: str) -> Iterable[Document]:
    loader = DirectoryLoader(
        dir_path, glob=glob_pattern, loader_cls=PyPDFLoader
    )  # Note: If you're using PyPDFLoader then it will split by page for you already
    documents = loader.load()
    logger.info(f"Loaded {len(documents)} documents from {dir_path}")
    return documents


def split(documents: Iterable[Document], chunk_size, chunk_overlap) -> List[Document]:
    """
    Splits the specified list of PyPDFLoader instances into text chunks using a recursive character text splitter.

    Args:
        documents  (Iterable[Document]): The documents to split.
        chunk_size (int): The size of each text chunk.
        chunk_overlap (int): The overlap between adjacent text chunks.

    Returns:
        List[str]: A list of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    logger.info(type(documents))
    logger.info(
        f"Splitting {len(documents)} documents into chunks of size {chunk_size} with overlap {chunk_overlap}"
    )
    logger.info(documents)
    texts = text_splitter.split_documents(documents)
    logger.info(f"Split {len(documents)} documents into {len(texts)}")
    return texts


def get_default_local_download(dir_path: str) -> List[Document]:
    """Default document list for local download"""
    glob_pattern = "*.pdf"

    chunks = load(dir_path, glob_pattern)
    texts = split(chunks, chunk_size=500, chunk_overlap=50)
    return texts
