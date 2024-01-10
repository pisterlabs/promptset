from langchain.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from typing import Any, Iterator, List, Mapping, Optional, Union
from langchain.document_loaders.blob_loaders import Blob
from langchain.document_loaders.parsers.pdf import (
    PyPDFParser
)

class BackPdfLoader(PyPDFLoader):
    """重写lazy_load方法，使得可以直接从byte类型的数据中加载pdf"""
    def __init__(self, bytes_data: bytes, file_path: str = None, password: str | bytes | None = None) -> None:
        # super().__init__(file_path, password)
        self.bytes_data = bytes_data
        self.parser = PyPDFParser(password=password)

    def lazy_load(
        self,
    ) -> Iterator[Document]:
        """Lazy load given path as pages."""
        blob = Blob.from_data(self.bytes_data)
        yield from self.parser.parse(blob)