import os
from abc import ABC
from typing import Any
from langchain.embeddings import GPT4AllEmbeddings, HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.base import BaseLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import GPT4All
from langchain.llms.base import LLM
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, VectorStore

from src.api import S3Connector
from src.config import Settings
from src.service.chatbot.loaders.s3_directory_loader import S3DirectoryLoader
from src.service.chatbot.chatbot import Chatbot

settings = Settings()


class BaseChatbot(Chatbot, ABC):

    llm: LLM
    vector_store: VectorStore | None
    chat_history: list[tuple[str, str]]

    def __init__(self, vector_store: VectorStore = None, llm: LLM = GPT4All(model=settings.LLM_PATH)):
        self.vector_store = vector_store
        self.llm = llm
        self.chat_history = []

    def load_vector_store(self, persist_directory: str, docs_path: str = None, replace: bool = False) -> None:

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        if os.listdir(persist_directory) and not replace:
            # load index
            db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        else:
            loader = S3DirectoryLoader(prefix=docs_path)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            texts = text_splitter.split_documents(documents)
            db = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory=persist_directory)
            db.persist()
        self.vector_store = db

    def chat(self, query: str, **kwargs) -> dict[str, Any]:

        llm = self.llm
        retriever = self.vector_store.as_retriever()
        qa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever,
                                                   return_source_documents=True)
        result = qa({"question": query, "chat_history": self.chat_history})
        self.chat_history.append((query, result["answer"]))
        return result



