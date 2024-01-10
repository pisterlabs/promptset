import os
from typing import List
from langchain.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
)

from .settings import settings

# FILE / TEXT UTILS


def scan_documents_folder(folder_path: str) -> List[str]:
    file_paths = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_paths.append(os.path.join(folder_path, filename))
    return file_paths


def preprocess_documents(folder_path: str = settings.documents_path):
    file_paths = scan_documents_folder(folder_path)
    if len(file_paths) == 0:
        raise Exception(f"No PDF's detected in: {folder_path}")

    documents = []
    for path in file_paths:
        print(f"Reading: {path}")
        loader = PyPDFLoader(path)
        documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)
    return chunks
