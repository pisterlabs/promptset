from langchain.document_loaders.base import BaseLoader
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd
from langchain.docstore.document import Document
from langchain import document_loaders
from pydantic import BaseModel
import logging
from umbertobot.git_utils import filter_gitignore_files

logger = logging.getLogger(__file__)


class PandasLoader(document_loaders.base.BaseLoader):
    """Loader that loads ReadTheDocs documentation directory dump."""

    def __init__(
        self,
        path: str,
        text_col: str,
        loader_type: str,
        included_cols: Optional[List[str]],
        errors: Optional[str] = None,
        **kwargs: Optional[Any]
    ):
        self.file_path = path
        self.errors = errors
        self.text_col = text_col
        self.load_file = pd.read_csv if loader_type == "csv" else pd.read_parquet
        self.included_cols = included_cols

    def load(self) -> List[Document]:
        """Load documents."""

        df = self.load_file(self.file_path)
        included_cols = (
            df.columns.drop(self.text_col)
            if self.included_cols is None
            else self.included_cols
        )
        df = df[included_cols]
        docs = []
        for __, row in df.iterrows():
            text = row[self.text_col]
            metadata = row.to_dict()
            metadata.pop(self.text_col)
            docs.append(Document(page_content=text, metadata=metadata))
        return docs


class GitignoreDirectoryLoader(document_loaders.base.BaseLoader):
    """
    like langchain directory loader but filters out files by .gitignore
    """

    def __init__(
        self,
        path: str,
        gitignore_path: str,
        glob: str = "**/[!.]*",
        silent_errors: bool = False,
        load_hidden: bool = False,
        loader_cls: document_loaders.directory.FILE_LOADER_TYPE = document_loaders.unstructured.UnstructuredFileLoader,
        recursive: bool = False,
    ):
        self.path = path
        self.glob = glob
        self.load_hidden = load_hidden
        self.loader_cls = loader_cls
        self.silent_errors = silent_errors
        self.recursive = recursive

    def load(self) -> List[Document]:
        """Load documents."""
        p = Path(self.path)
        docs = []
        items = p.rglob(self.glob) if self.recursive else p.glob(self.glob)
        items = filter_gitignore_files(items, self.gitignore_path)
        for i in items:
            if i.is_file():
                if (
                    document_loaders.directory._is_visible(i.relative_to(p))
                    or self.load_hidden
                ):
                    try:
                        sub_docs = self.loader_cls(str(i)).load()
                        docs.extend(sub_docs)
                    except Exception as e:
                        if self.silent_errors:
                            logger.warning(e)
                        else:
                            raise e
        return docs


def get_directory_loader(path, glob="**/[!.]*", gitignore_path=None):
    if gitignore_path is None:
        return document_loaders.DirectoryLoader(path, glob)
    else:
        return GitignoreDirectoryLoader(path, glob, gitignore_path)
