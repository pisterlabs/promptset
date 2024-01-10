from Loader.BaseLoader import BaseLoader
from Loader.LocalDocumentLoader import LocalDocumentLoader
import os
from typing import List
from langchain.docstore.document import Document
from tqdm import tqdm


class LocalDirectoryLoader(BaseLoader):

    def __init__(self, dir_path):
        super().__init__()
        self.document_loader = None
        self.dir_path = dir_path

    def load_dir(self):
        # Loads all documents from source documents directory
        print(f"Loading documents from {self.dir_path}")
        all_files = os.listdir(self.dir_path)

        documents = []
        for file_path in tqdm(all_files, desc="Loading documents"):
            if file_path[-4:] in ['.txt', '.pdf', '.csv']:
                self.document_loader = LocalDocumentLoader(os.path.join(self.dir_path, file_path))
                self.document_loader.load_document()
                self.document_loader.preprocess_data()
                documents.append(self.document_loader.load())

        self.data = documents
        print(f"Loaded {len(documents)} documents from {self.dir_path}")

    def load(self) -> List[Document]:
        self.load_dir()
        return self.data
