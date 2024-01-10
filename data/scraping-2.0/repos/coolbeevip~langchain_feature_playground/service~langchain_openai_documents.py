# -*- coding: UTF-8 -*-
import os

from langchain import FAISS
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

from service.langchain_openai_interface import ILangChainOpenAI


class LangChainOpenAIDocuments(ILangChainOpenAI):
    def __init__(self):
        file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'state_of_the_union.txt'))
        loader = TextLoader(file_path)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)
        self.embeddings = OpenAIEmbeddings()
        self.db = FAISS.from_documents(docs, self.embeddings)

    def get_answer(self, question: str) -> str:
        embedding_vector = self.embeddings.embed_query(question)
        docs = self.db.similarity_search_by_vector(embedding_vector)
        return docs[0].page_content
