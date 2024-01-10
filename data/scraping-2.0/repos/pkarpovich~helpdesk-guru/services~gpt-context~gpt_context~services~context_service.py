import os
from typing import TYPE_CHECKING

from langchain.document_loaders import TextLoader, PDFMinerLoader, CSVLoader, GoogleDriveLoader
from langchain.document_loaders.base import BaseLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from gpt_context.adapters import VectorStoreAdapter

if TYPE_CHECKING:
    from gpt_context.services import AppConfig


class ContextService:
    def __init__(self, config: 'AppConfig', store: VectorStoreAdapter):
        self.store = store
        self.config = config

    def init(self) -> None:
        for root, _, files in os.walk("source_documents"):
            if not files:
                print(f"The directory '{root}' is empty.")
                return

            for file in files:
                loader = self._get_file_loader(root, file)

                if loader is None:
                    print("No loader found")
                    return

                documents = loader.load()
                self._add_documents_to_store(documents)

    def clear_index(self, context_name) -> None:
        self.store.truncate(context_name)

    def add_google_docs(self, folder_id: str, context_name: str) -> None:
        loader = self._prepare_google_drive_loader(folder_id)
        docs = loader.load()
        self._add_documents_to_store(docs, context_name)

    def check_google_docs_connectivity(self, folder_id: str) -> bool:
        loader = self._prepare_google_drive_loader(folder_id)
        docs = loader.load()

        return len(docs) > 0

    @staticmethod
    def _get_file_loader(root: str, file: str) -> BaseLoader:
        ext = os.path.splitext(file)[-1].lower()
        loader = None

        match ext:
            case ".txt":
                loader = TextLoader(os.path.join(root, file), encoding="utf8")
            case ".pdf":
                loader = PDFMinerLoader(os.path.join(root, file))
            case ".csv":
                loader = CSVLoader(os.path.join(root, file))

        return loader

    def _add_documents_to_store(self, documents: list[Document], context_name: str) -> None:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=200, separators=["Question"])
        texts = text_splitter.split_documents(documents)

        self.store.from_documents(texts, index_name=context_name)

    def _prepare_google_drive_loader(self, folder_id: str) -> GoogleDriveLoader:
        return GoogleDriveLoader(
            credentials_path=self.config.GOOGLE_APPLICATION_CREDENTIALS,
            token_path=self.config.GOOGLE_APPLICATION_TOKEN,
            folder_id=folder_id,
            recursive=False,
        )
