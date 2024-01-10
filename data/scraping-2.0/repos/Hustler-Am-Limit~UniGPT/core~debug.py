from langchain.vectorstores.base import VectorStore
from typing import Iterable, List, Any, Optional
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.embeddings.fake import FakeEmbeddings as FakeEmbeddingsBase
from langchain.chat_models.fake import FakeListChatModel


class FakeChatModel(FakeListChatModel):
    def __init__(self, **kwargs):
        responses = ["""👈 You first need to enter your secret code into the text box on the left side!

Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet."""]
        super().__init__(responses=responses, **kwargs)


class FakeEmbeddings(FakeEmbeddingsBase):
    def __init__(self, **kwargs):
        super().__init__(size=4, **kwargs)


class FakeVectorStore(VectorStore):
    """Fake vector store for testing purposes."""

    def __init__(self, texts: List[str]):
        self.texts: List[str] = texts

    def add_texts(
        self, texts: Iterable[str], metadatas: List[dict] | None = None, **kwargs: Any
    ) -> List[str]:
        self.texts.extend(texts)
        return self.texts

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> "FakeVectorStore":
        return cls(texts=list(texts))

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        return [
            Document(page_content=text, metadata={"source": f"{i+1}-{1}"})
            for i, text in enumerate(self.texts)
        ]
