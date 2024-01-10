"""Loading logic for loading documents from a directory."""
import logging
from pathlib import Path
from typing import List, Type, Union

from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders.html_bs import BSHTMLLoader
from langchain.document_loaders.text import TextLoader
from langchain.document_loaders.unstructured import UnstructuredFileLoader

FILE_LOADER_TYPE = Union[
    Type[UnstructuredFileLoader], Type[TextLoader], Type[BSHTMLLoader]
]
logger = logging.getLogger(__name__)


def _is_visible(p: Path) -> bool:
    parts = p.parts
    for _p in parts:
        if _p.startswith("."):
            return False
    return True


class MappingDirectoryLoader(BaseLoader):
    """Loading logic for loading documents from a directory."""

    def __init__(
            self,
            path: str,
            glob: str = "**/[!.]*",
            silent_errors: bool = False,
            load_hidden: bool = False,
            loader_mapping=None,  # function taking in file extension and returning loader class and kwargs
            recursive: bool = False,
    ):
        """Initialize with path to directory and how to glob over it."""
        # loader mapping maps file extensions to loader classes and loader kwargs
        if loader_mapping is None:
            # TODO: expand mapping to include more document loaders
            #  from https://python.langchain.com/en/latest/modules/indexes/document_loaders.html
            def loader_mapping(ext: str):
                if ext in (".txt", ".py"):
                    return TextLoader, {"encoding": "utf-8"}
                elif ext in (".html", '.htm'):
                    return BSHTMLLoader, {}
                elif ext == ".pdf":
                    return PyPDFLoader, {}
                else:
                    return TextLoader, {"encoding": "utf-8"}

        self.path = path
        self.glob = glob
        self.load_hidden = load_hidden
        self.loader_mapping = loader_mapping
        self.silent_errors = silent_errors
        self.recursive = recursive

    def load(self) -> List[Document]:
        """Load documents."""
        p = Path(self.path)
        docs = []
        items = p.rglob(self.glob) if self.recursive else p.glob(self.glob)
        for i in items:
            if i.is_file():
                if _is_visible(i.relative_to(p)) or self.load_hidden:
                    try:
                        loader_cls, loader_kwargs = self.loader_mapping(i.suffix)
                        sub_docs = loader_cls(str(i), **loader_kwargs).load()
                        docs.extend(sub_docs)
                    except Exception as e:
                        if self.silent_errors:
                            logger.warning(e)
                        else:
                            raise e
        return docs
