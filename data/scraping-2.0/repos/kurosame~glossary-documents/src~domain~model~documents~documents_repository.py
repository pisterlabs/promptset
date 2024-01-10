from abc import ABC, abstractmethod

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.vectorstores import SupabaseVectorStore


class DocumentsRepository(ABC):
    @abstractmethod
    def from_with_query(
        self, docs: list[Document], query_name: str
    ) -> SupabaseVectorStore:
        raise NotImplementedError()

    def delete_all(self) -> None:
        raise NotImplementedError()
