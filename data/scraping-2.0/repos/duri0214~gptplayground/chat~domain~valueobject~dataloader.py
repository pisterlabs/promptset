from abc import ABC, abstractmethod
from typing import List

from langchain.schema import Document
from langchain.text_splitter import TokenTextSplitter


class Dataloader(ABC):
    @property
    @abstractmethod
    def data(self) -> List[Document]:
        pass

    @abstractmethod
    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 100):
        self.text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    @abstractmethod
    def _load(self):
        pass
    
    @abstractmethod
    def _split(self):
        pass

    def _shredder(self, text: str, source: str, attr: str) -> tuple:
        """
        日本語PDFでトークンを多く消費するような場合、ページ単位ではAPIが処理できないので
        さらに千切りにする
        """
        text_fragments = self.text_splitter.split_text(text)
        all_text, all_metadata = [], []
        for text_fragment in text_fragments:
            all_text.extend(text_fragment)
            all_metadata.extend({"source": source, "attr": attr})

        return all_text, all_metadata
