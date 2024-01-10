from typing import Self

from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

from camila.settings import Settings


class Store:
    COLLECTION_NAME = "camila"

    def __init__(self: Self, settings: Settings) -> None:
        self._vectorstore = Chroma(
            collection_name=self.COLLECTION_NAME,
            embedding_function=OpenAIEmbeddings(openai_api_key=settings.OPENAI_API_KEY),
            persist_directory=str(settings.STORE_PERSIST_DIRECTORY_PATH),
        )

    @property
    def retriever(self: Self) -> Chroma:
        return self._vectorstore.as_retriever()

    def add_documents(self: Self, documents: list[Document]) -> None:
        self._vectorstore.add_documents(documents=documents)
