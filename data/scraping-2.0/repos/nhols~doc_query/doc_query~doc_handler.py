from typing import Type
from langchain.document_loaders import TextLoader, UnstructuredEPubLoader, UnstructuredPDFLoader, OnlinePDFLoader
from langchain.document_loaders.base import BaseLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from doc_query.app_config import config


class DocHandler:
    loader_map: dict[str, Type[BaseLoader]] = {
        "pdf": UnstructuredPDFLoader,
        "epub": UnstructuredEPubLoader,
        "online_pdf": OnlinePDFLoader,
    }

    def __init__(self, filename: str, doc_name: str):
        self.filename = filename
        self.doc_name = doc_name
        self.loader: Type[BaseLoader] = self.get_loader()
        self._doc = None
        self._doc = self.doc

    @property
    def loader_key(self) -> str:
        if self.filename.startswith("http") and self.filename.endswith(".pdf"):
            return "online_pdf"
        return self.filename.split(".")[-1]

    def get_loader(self) -> Type[BaseLoader]:
        return self.loader_map.get(self.loader_key, TextLoader)

    @property
    def doc(self) -> list[Document]:
        if self._doc is None:
            self._doc = self.load_doc()
        return self._doc

    def load_doc(self) -> list[Document]:
        return self.loader(self.filename).load()

    def split_doc(self) -> list[Document]:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        return text_splitter.split_documents(self.doc)

    def split_embed_doc(self) -> None:
        split_docs = self.split_doc()
        config.vectorstore.add_texts(
            texts=[split_doc.page_content for split_doc in split_docs],
            namespace=self.doc_name,
        )
