"""
TODO: Provide a description for this file/module here.
"""
from abc import ABC, abstractmethod
from typing import List

# Ignore this import error, it's a bug in VSCode
# pylint: disable=import-error
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.crawlers.data_structures.article import Article


# pylint: disable=too-few-public-methods
class ChunkService(ABC):
    """
    Abstract base class for chunking services.

    Provides an interface for services that split textual content into smaller chunks.
    """

    @abstractmethod
    def split_into_chunks(self, article: Article) -> List[str]:
        """
        Split the given article into smaller textual chunks.

        Parameters:
            article (Article): The article to be split.

        Returns:
            List[str]: A list of textual chunks.
        """


# pylint: disable=too-few-public-methods
class ContentAwareChunkingService(ChunkService):
    """
    Content-aware chunking service.

    This service implements a content-aware chunking strategy,
    splitting the content based on its structure.
    """

    def split_into_chunks(self, article: Article) -> List[str]:
        """
        Split the given article into chunks based on its content structure.

        For this basic implementation, the article text is split by paragraphs.

        Parameters:
            article (Article): The article to be split.

        Returns:
            List[str]: A list of textual chunks.
        """
        # Split the article text by paragraphs
        chunks = article.text.split("\n\n")
        # Remove any empty strings from the list
        return [chunk for chunk in chunks if chunk]


# pylint: disable=too-few-public-methods
class LangChainChunkingService(ChunkService):
    """
    Chunking service using the langchain text_splitter.

    This service uses the RecursiveCharacterTextSplitter from
    langchain to split the content into chunks.
    """

    def __init__(self, chunk_size: int, chunk_overlap: int):
        """
        Initialize the LangChainChunkingService with the desired chunk size and overlap.

        Parameters:
            chunk_size (int): The desired size for each chunk.
            chunk_overlap (int): The overlap between consecutive chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_into_chunks(self, article: Article) -> List[str]:
        """
        Split the given article into chunks using the langchain text_splitter.

        Parameters:
            article (Article): The article to be split.

        Returns:
            List[str]: A list of textual chunks.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            add_start_index=True,
        )

        # Use the article's text for splitting
        documents = text_splitter.create_documents([article.text])

        # Extract the actual text from the Document objects
        texts = [doc.page_content for doc in documents]

        return texts
