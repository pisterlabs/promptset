from abc import ABC, abstractmethod
from typing import Union, List
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


class Loaders(ABC):
    """Interface for all loaders."""

    @abstractmethod
    def load(self, path: str) -> Union[str, List[str], List[Document]]:
        """
        Load content from a file and return it.

        :param path: The path to the file to be loaded.
        :return: The content of the file. Either as String, as List of Strings or as List of Langchain Documents.
        """
        pass

    def load_file(self, path: str) -> str:
        """Default implementation using 'with' for file handling."""
        with open(path, 'r') as file:
            content = file.read()
        return content

    @abstractmethod
    def chunkDocument(self, document, chunkSize) -> List[Document]:
        """Chunk a document into smaller parts."""
        pass

    def defaultChunker(self, document, chunkSize) -> List[Document]:
        """Default implementation using RecursiveCharacterTextSplitter"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunkSize,
            chunk_overlap=20,
            length_function=len,
            add_start_index=True,
        )
        chunked_content= text_splitter.create_documents(document)
        return chunked_content
