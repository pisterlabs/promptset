from pathlib import Path
from typing import Self

from langchain.docstore.document import Document
from langchain.document_loaders import DirectoryLoader, UnstructuredFileLoader


class Loader:
    """
    Loads data from a source to a list of documents.

    A `Document` is a piece of text and associated metadata.
    """

    def __init__(self: Self) -> None:
        self._docs = []

    @property
    def documents(self: Self) -> list[Document]:
        return self._docs

    def load_directory(self: Self, path: Path | str, glob: str) -> bool:
        """
        Loads all files in a directory.

        Returns:
            bool: Whether all files were loaded successfully.
        """

        loader = DirectoryLoader(path, glob=glob, loader_cls=UnstructuredFileLoader)
        self._docs.extend(loader.load())
        return True

    def load_file(self: Self, path: Path | str) -> bool:
        """
        Loads a single file.

        Returns:
            bool: Whether the file was loaded successfully.
        """

        loader = UnstructuredFileLoader(path)
        self._docs.extend(loader.load())
        return True
