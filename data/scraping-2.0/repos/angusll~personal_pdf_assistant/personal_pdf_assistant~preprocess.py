import os
from typing import List, Union

from loguru import logger
from models import Credentials
from tqdm import tqdm

from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

creds = Credentials()


def config_text_splitter(
    chunk_size: int = 512,
    chunk_overlap: int = 20,
    length_function: str = "len",
    add_start_index: bool = True,
) -> RecursiveCharacterTextSplitter:
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=length_function,
        add_start_index=add_start_index,
    )
    return text_splitter


def preprocess_pdf(
    pdf_path: Union[str, os.PathLike],
    text_splitter: RecursiveCharacterTextSplitter,
) -> List[Document]:
    """
    Load and split a single PDF file
    """
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load_and_split(text_splitter)
    return pages


def preprocess_pdf_from_directory(
    text_splitter: RecursiveCharacterTextSplitter,
    dir_path: Union[str, os.PathLike],
    glob: str = "**/*.pdf",
    use_multithreading: bool = False,
    max_concurrency: int = 4,
    show_progress: bool = True,
) -> List[Document]:
    """
    Load and split PDF files from a directory
    """
    loader = DirectoryLoader(
        str(dir_path),
        glob=glob,
        use_multithreading=use_multithreading,
        max_concurrency=max_concurrency,
        show_progress=show_progress,
    )
    logger.info("Loading PDFs")
    pages = loader.load_and_split(text_splitter)
    return pages


def get_pdf_embeddings(
    pages: List[Document], embedding_model_name: str = "ada"
) -> List[List[float]]:
    # TODO add multi thread to speed up

    embeddings = OpenAIEmbeddings(
        model=embedding_model_name, openai_api_key=creds.openai_api_key
    )
    # Turn the first text chunk into a vector with the embedding[
    logger.info("Getting embeddings")
    embeds = [
        embeddings.embed_query(page.page_content) for page in tqdm(pages)
    ]
    return embeds
