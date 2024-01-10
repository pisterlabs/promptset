from pathlib import Path
from typing import List

import tqdm
from langchain.document_loaders import TextLoader
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter

from kb_guardian.logger import INFO_LOGGER


def load_text_documents(file_path: str) -> List[Document]:
    """
    Return a list of LangChain Documents from a text file.

    Args:
    path (str): Path to a text document file.

    Returns:
    List[Document]: A list of LangChain Documents with page_content and
                    extra info about as the source

    """
    loader = TextLoader(file_path, encoding="utf-8")
    document = loader.load()
    for doc in document:
        doc.metadata["source"] = Path(file_path).stem
    return document


def split_documents(
    docs: List[Document], chunk_size: int, chunk_overlap: int
) -> List[Document]:
    """
    Split the given list of LangChain Documents into smaller chunks.

    Args:
        docs (List[Document]): A list of LangChain Documents to be split.
        chunk_size (int): The desired size of each chunk.
        chunk_overlap (int): The number of overlapping characters in adjacent chunks.

    Returns:
        List[Document]: A list of LangChain Documents containing the split chunks.

    """
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator=".",
        add_start_index=True,
    )
    splitted_docs = text_splitter.split_documents(docs)
    return splitted_docs


def create_document_chunks(
    data_path: str,
    extension: str,
    chunk_size: int,
    chunk_overlap: int,
) -> List[Document]:
    """
    Convert documents present in the data source into a list of LangChain Documents, then split those Documents into smaller chunks, digestible by the LLM.

    Args:
        data_path (str): The path to the data source.
        extension (str): The file extension of the data sources.
        chunk_size (int): The desired size of each chunk.
        chunk_overlap (int): The number of overlapping characters in adjacent chunks.

    Returns:
        List[Document]: A list of document chunks created from the data sources.

    """  # noqa: E501
    documents = []

    files = [str(p.resolve()) for p in Path(data_path).glob(f"**/*.{extension}")]
    for file in tqdm.tqdm(files):
        try:
            doc = load_text_documents(file)
            documents.extend(doc)
        except Exception as e:
            INFO_LOGGER.warning(
                f"{str(e)} Document conversion failed for file '{Path(file).stem}': "
            )

    return split_documents(documents, chunk_size, chunk_overlap)
