import typing as t
from functools import cached_property

from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.base import \
    BaseConversationalRetrievalChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.vectorstores.base import VectorStoreRetriever

from src.generative_ai.large_language_models.chatbots import Chatbot, ModelArgs


class ChatbotRAG(Chatbot):
    def __init__(
        self,
        vector_store: FAISS | None = None,
        embeddings_kwargs: t.Dict | None = None,
        search_kwargs: t.Dict | None = None,
        **model_kwargs: t.Unpack[ModelArgs],
    ) -> None:
        super().__init__(**model_kwargs)
        if vector_store:
            self.vector_store = vector_store
        self.embeddings_kwargs = embeddings_kwargs or {}
        self.search_kwargs = search_kwargs or {}

    @cached_property
    def embeddings(self) -> OpenAIEmbeddings:
        return OpenAIEmbeddings(**self.embeddings_kwargs)

    @cached_property
    def vector_store(self) -> FAISS:
        return FAISS.load_local(folder_path="faiss_index", embeddings=self.embeddings)

    @cached_property
    def retriever(self) -> VectorStoreRetriever:
        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs=self.search_kwargs,
        )

    @cached_property
    def chain(self) -> BaseConversationalRetrievalChain:
        return ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            memory=self.memory,
            verbose=True,
            combine_docs_chain_kwargs={"prompt": self.template},
            chain_type="stuff",
            retriever=self.retriever,
        )

    def ask(
        self,
        query: str,
        language: str | None = None,
    ) -> str:
        return self.chain.run(
            question=query,
            language=language or "the input language",
            callbacks=self.callbacks,
        )
