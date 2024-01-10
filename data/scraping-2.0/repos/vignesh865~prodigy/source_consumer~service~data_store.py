from abc import ABC, abstractmethod
from typing import List

from langchain.schema import Document


class DataStore(ABC):

    @abstractmethod
    def get_store(self, collection_name):
        pass

    @abstractmethod
    def update_data(self, collection_name: str, chunked_docs: List[Document]):
        pass

    @abstractmethod
    def get_documents(self, collection_name):
        pass
