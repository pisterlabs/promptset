import os
from typing import Iterator

from langchain.schema import Document
from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader

from .preprocess_data import make_texts_tokenisation_safe


class DataLoader(BaseLoader):
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.metadata = {}

    def lazy_load(self) -> Iterator[Document]:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement lazy_load()"
        )

    def load(self):
        documents = list(self.lazy_load())
        self.metadata.update({"num_documents": len(documents)})
        return documents


class PDFDataLoader(DataLoader):
    def __init__(self, data_dir: str):
        super().__init__(data_dir)
        self.metadata = {
            "data_dir": data_dir,
            "loader": "PDFDataLoader",
            "num_documents": None,
        }

    @make_texts_tokenisation_safe
    def lazy_load(self) -> Iterator[Document]:
        try:
            # document = DirectoryLoader(
            #     self.data_dir, glob="*.pdf", loader_cls=PyPDFLoader
            # ).load()
            document = PyPDFLoader(self.data_dir).load()
            for doc in document:
                doc.metadata["file_type"] = os.path.splitext(doc.metadata["source"])[-1]
            return document

        except Exception as e:
            print(f"Error loading : {e}")
            return None


class TextDataLoader(DataLoader):
    def __init__(self, data_dir: str):
        super().__init__(data_dir)
        self.metadata = {
            "data_dir": data_dir,
            "loader": "TextDataLoader",
            "num_documents": None,
        }

    @make_texts_tokenisation_safe
    def lazy_load(self) -> Iterator[Document]:
        try:
            document = DirectoryLoader(
                self.data_dir, glob="*.txt", loader_cls=TextLoader
            ).load()
            for doc in document:
                doc.metadata["file_type"] = os.path.splitext(doc.metadata["source"])[-1]
            return document

        except Exception as e:
            print(f"Error loading : {e}")
            return None


# pdf_loader = PDFDataLoader(data_dir="E:/Projects/Hackathons/StudyBot/data")

# documents = pdf_loader.load()
# print(documents)
