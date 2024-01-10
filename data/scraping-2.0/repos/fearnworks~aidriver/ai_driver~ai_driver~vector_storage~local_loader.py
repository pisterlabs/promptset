from typing import List, Iterable, Dict
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader, UnstructuredMarkdownLoader, JSONLoader
from loguru import logger
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.storage import RedisStore
from langchain.embeddings import CacheBackedEmbeddings
import os 
import json 
import PyPDF2

def flatten_docs(list_of_docs):
    flat_list = []
    for item in list_of_docs:
        if isinstance(item, list):
            flat_list.extend(flatten_docs(item))
        else:
            flat_list.append(item)
    return flat_list


def directory_to_docs(dir_path:str) -> List[Document]:
    docs = []
    logger.info('Loading PDF')
    docs.append(load(dir_path, "*.pdf", PyPDFLoader))
    logger.info('Loading txt')
    docs.append(load(dir_path, "*.txt", TextLoader))
    logger.info('Loading markdown')
    docs.append(load(dir_path, "*.md", UnstructuredMarkdownLoader))
    # docs.append(load(dir_path, "*.json", JSONLoader, loader_kwargs = {'jq_schema':'.content'}))
    flattened_docs = flatten_docs(docs)
    return flattened_docs


def load(dir_path: str, glob_pattern: str, loader, loader_kwargs = None) -> Iterable[Document]:
    loader = DirectoryLoader(
        dir_path, glob=glob_pattern, loader_cls=loader, show_progress=True, loader_kwargs=loader_kwargs
    )  # Note: If you're using PyPDFLoader then it will split by page for you already
    documents = loader.load()
    logger.info(f"Loaded {len(documents)} documents from {dir_path}")
    return documents

def load_text_to_doc(dir_path: str, glob_pattern: str) -> Iterable[Document]:
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

def get_cache_embeddings(
    embedding_model_name: str, embedding_model_kwargs: Dict[str, str]
) -> CacheBackedEmbeddings:
    store = RedisStore(
        redis_url="redis://ai_driver_redis:26379", namespace="embedding_caches"
    )
    underlying_embeddings = HuggingFaceInstructEmbeddings(
        model_name=embedding_model_name,
        model_kwargs=embedding_model_kwargs,
        cache_folder="",
    )
    embedder = CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings, store, namespace=embedding_model_name
    )
    return embedder

def load_files_from_directory(directory_path: str)-> List[str]:
    all_texts = []
    for filename in os.listdir(directory_path):
        full_path = os.path.join(directory_path, filename)
        if filename.endswith('.txt'):
            with open(full_path, 'r', encoding='utf-8') as f:
                all_texts.append(f.read())
        elif filename.endswith('.json'):
            with open(full_path, 'r', encoding='utf-8') as f:
                all_texts.append(json.load(f))
        elif filename.endswith('.md'):
            with open(full_path, 'r', encoding='utf-8') as f:
                all_texts.append(f.read())
        elif filename.endswith('.pdf'):
            with open(full_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                content = []
                for i in range(len(reader.pages)):
                    content += reader.pages[i].extract_text().split()
                all_texts.append(' '.join(content))
    return all_texts

def get_default_local_download(dir_path: str) -> List[Document]:
    """Default document list for local download"""
    chunks = directory_to_docs(dir_path)
    texts = split(chunks, chunk_size=500, chunk_overlap=50)
    return texts
