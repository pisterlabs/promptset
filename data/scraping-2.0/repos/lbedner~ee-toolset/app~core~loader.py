import os

from langchain.document_loaders import (
    PDFPlumberLoader,
    PythonLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    UnstructuredRTFLoader,
    UnstructuredWordDocumentLoader,
)
from langchain.document_loaders.base import BaseLoader
from langchain.schema import Document

from app.core.log import logger
from app.models import KnowledgeBaseDocument


def load_file_from_file_name(
    file_name: str,
    file_path: str,
    file_size: int,
    loader_class: BaseLoader,
    file_uri: str = None,
) -> list[Document]:
    loader: BaseLoader = loader_class(file_path)
    loaded_data = loader.load()

    # We want to set source to a URI if it exists, otherwise the file name
    source: str = file_uri if file_uri else file_name
    loaded_data[0].metadata["source"] = source
    loaded_data[0].metadata["size"] = file_size
    return loaded_data


def determine_loader_and_load(
    file_name: str,
    file_path: str,
    file_size: int,
    file_uri: str = None,
) -> list[Document]:
    ext = os.path.splitext(file_name)[1].lower()

    loader_map = {
        ".pdf": PDFPlumberLoader,
        ".html": UnstructuredHTMLLoader,
        ".htm": UnstructuredHTMLLoader,
        ".rtf": UnstructuredRTFLoader,
        ".txt": TextLoader,
        ".text": TextLoader,
        ".doc": UnstructuredWordDocumentLoader,
        ".docx": UnstructuredWordDocumentLoader,
        ".py": PythonLoader,
        ".json": TextLoader,
        ".sql": TextLoader,
    }

    loader_class = loader_map.get(ext)
    if loader_class:
        return load_file_from_file_name(
            file_name=file_name,
            file_path=file_path,
            file_size=file_size,
            loader_class=loader_class,
            file_uri=file_uri,
        )
    else:
        logger.error("loader.not.found", file_name=file_name, ext=ext)
        return []


def load_documents(
    knowledge_base_documents: dict[str, KnowledgeBaseDocument]
) -> list[Document]:
    documents: list[Document] = []
    for knowledge_base_document in knowledge_base_documents.values():
        file_name: str = os.path.basename(knowledge_base_document.Filepath)
        logger.debug("loader.load_documents", file_name=file_name)
        documents.extend(
            determine_loader_and_load(
                file_name=file_name,
                file_path=knowledge_base_document.Filepath,
                file_size=knowledge_base_document.Size,
                file_uri=knowledge_base_document.Uri,
            )
        )
    return documents
