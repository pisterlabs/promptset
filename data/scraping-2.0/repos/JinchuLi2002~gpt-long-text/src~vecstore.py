from langchain.embeddings import OpenAIEmbeddings
import os
from langchain.vectorstores import FAISS


class VectorStore:
    def __init__(
        self,
    ):
        self.embedding = OpenAIEmbeddings()
        self.db = FAISS.from_texts(
            [''], self.embedding)

    def clear(self):
        self.db = FAISS.from_texts(
            [''], self.embedding)

    def add_texts(self, texts: list[str]):
        self.db.add_texts(texts)

    def add_docs(self, docs):
        self.db.add_documents(docs)
