from __future__ import annotations
from langchain.document_loaders import PyPDFLoader, UnstructuredFileLoader
from langchain.document_loaders.base import BaseLoader
from utilities import get_file_extension


class FileLoaderFactory:

    @staticmethod
    def get_loader(file_path: str) -> BaseLoader:
        """Get loader for file path."""
        file_extension = get_file_extension(file_path)
        if file_extension == ".pdf":
            return PyPDFLoader(file_path)
        else:
            return UnstructuredFileLoader(file_path)

