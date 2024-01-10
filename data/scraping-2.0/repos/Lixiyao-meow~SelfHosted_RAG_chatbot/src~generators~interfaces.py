from typing import Callable, Protocol, List
from langchain.schema.embeddings import Embeddings
from langchain.schema import Document

from llama_cpp.llama_chat_format import ChatFormatterResponse

class SimpleEmbeddingModel(Protocol):
    def __call__(self, text: str) -> List[float]:
        ...

class EmbeddingModel(Embeddings):

    def __init__(self, simple_function: Callable[[str], List[float]]) -> None:
        super().__init__()
        self.embedding_function = simple_function

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embedding_function(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self.embedding_function(text)


class SimpleGenerativeModel(Protocol):
    def RAG_QA_chain(self, retrieved_docs: List[Document], query: str) -> str | tuple[list[str], str]:
        ...