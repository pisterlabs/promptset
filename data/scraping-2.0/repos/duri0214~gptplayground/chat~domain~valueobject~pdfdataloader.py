import os
from typing import List

from langchain.document_loaders import PyPDFLoader
from langchain.schema import Document

from chat.domain.valueobject.dataloader import Dataloader


class PdfDataloader(Dataloader):
    @property
    def data(self) -> List[Document]:
        return self.pages

    def __init__(self, file_path: str):
        super().__init__()
        self._file_path = file_path
        self._load()
        self._split()

    def _load(self):
        self.pages = PyPDFLoader(self._file_path).load()

    def _split(self):
        """
        PDFを切り刻み、出典（ページ数）をつけます
        """
        filename = os.path.basename(self._file_path)
        for i, doc in enumerate(self.pages):
            doc.page_content = doc.page_content.replace("\n", " ")
            doc.metadata = {"source": f'{filename} {i + 1}ページ'}
