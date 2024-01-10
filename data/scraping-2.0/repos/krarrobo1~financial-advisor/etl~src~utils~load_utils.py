from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, PyMuPDFLoader
from src.utils.storage_utils import list_folders
from glob import glob
from tqdm import tqdm
import logging


def get_vectorizer(model_name="multi-qa-MiniLM-L6-cos-v1"):
    """
    Create and return a Sentence Transformer Embeddings model using the specified model name.

    Args:
        model_name (str): The name if the Sentence Transformer model to use. Default is "multi-qa-MiniLM-L6-cos-v1"

    Returns:
        SentenceTransformerEmbeddings: An instance of the Sentence Transformer model.
    """
    return SentenceTransformerEmbeddings(model_name=model_name)


def load_documents(directory: str):
    """
    Load and split text from PDF documents in the specified directory.

    Args:
        directory (str): The path to the directory containig PDF files.

    Returns:
        list: A list of text documents extracted from the PDFs
    """

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=250)
    documents = []
    for item_path in tqdm(glob(directory + "*.pdf")):
        logging.info(f"Attemping to load {item_path}")
        try:
            loader = PyPDFLoader(item_path)
            documents.extend(loader.load_and_split(text_splitter=text_splitter))
        except Exception as e:
            logging.error(f"load_documents: Could not load {item_path}")
            continue

    return documents


def load_documents_directory(directory):
    """
    Load and split text from documents in the specified directory using multi-threading.

    Args:
        directory (str): The path to the directory containing documents.

    Returns:
        list: A list of text documents extracted from the directory
    """

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=40)
    documents = []
    loader = DirectoryLoader(
        directory, use_multithreading=True, silent_errors=True, show_progress=True
    )
    documents.extend(loader.load_and_split(text_splitter=text_splitter))

    return documents


def load_documents_2(directory):
    """
    Load and split text from PDF documents in the specified directory using PyMuPDF.

    Args:
        directory (str): The path to the directory containing PDF files.

    Returns:
        list: A list of text documents extracted from the PDFs.
    """

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=25)
    documents = []
    for item_path in tqdm(glob(directory + "*.pdf")):
        try:
            loader = PyMuPDFLoader(item_path)
            documents.extend(loader.load_and_split(text_splitter=text_splitter))
        except:
            continue

    return documents


def load_db(embedding_function, persistence_directory="chroma_db-database"):
    """
    Create and return a Chroma database using the specified embedding function.

    Args:
        embedding_function (callable): The function used to compute document embeddings.
        persistence_directory (str): The directory where the Chroma database will be persisted.

    Returns:
        Chroma: An instance of the Chroma database.
    """

    db = Chroma(
        persist_directory=persistence_directory, embedding_function=embedding_function
    )
    return db


def split_docs_in_batches(input_list, batch_size):
    """
    Split a list of documents into batches of the specified size.

    Args:
        input_list (list): The list of documents to be split.
        batch_size (int): The size of each batch

    Yields:
        list: A batch of documents.
    """

    for i in range(0, len(input_list), batch_size):
        yield input_list[i : i + batch_size]


def save_db(documents, embedding_function, persistence_directory="chroma_db-database"):
    """
    Create a Chroma database from a list of documents and save it to the specified directory.

    Args:
        documents (list): A list of text documents.
        embedding_function (callable): The function used to compute document embeddings.
        persistence_directory (str): The directory where the Chroma database will be persisted. Default is "chroma_db-database"
    """
    docs_batches = split_docs_in_batches(documents, 3500)

    for doc_batch in tqdm(docs_batches):
        db = Chroma.from_documents(
            documents=doc_batch,
            embedding=embedding_function,
            persist_directory=persistence_directory,
        )

        db.persist()
