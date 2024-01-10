# coding=utf-8

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.document import Document
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.vectorstores.base import VectorStore


from typing import List

class RetrieverBuilder(object):
    def build_retriever(self, vector_store: VectorStore):
       # vector_store = Chroma.from_documents(documents=docs, embedding=OpenAIEmbeddings())
        # user multi-query retriever
        retriever_from_llm = MultiQueryRetriever.from_llm(retriever=vector_store.as_retriever(), llm=ChatOpenAI(temperature=0))

        return vector_store.as_retriever()
