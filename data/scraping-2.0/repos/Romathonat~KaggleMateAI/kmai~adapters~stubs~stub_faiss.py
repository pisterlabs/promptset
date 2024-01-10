from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document

from kmai.ports.ivectorstore_helper import IVectorStoreHelper
import pandas as pd

class StubFAISSFromScratch(IVectorStoreHelper):
    def create_vectorstore(self, docs):
        pass

    def read_vectorstore(self, path: str):
        pass

    def write_vectorstore(self, db: FAISS, path: str):
        pass 

    def similarity_search(self, db: FAISS, text: str, k: int):
        output = []
        for i in range(k):
            output.append(Document(page_content="foo", metadata={
                "Title": f"Title{i}",
                "Description": f"Description{i}",
                "Url": f"https://www.kaggle.com/c/{i}",
            }))
        return output

    def add_doc_to_vectorstore(self, db, docs):
        pass

class StubFAISSExistingData(IVectorStoreHelper):
    def create_vectorstore(self, docs):
        pass

    def read_vectorstore(self, path: str):
        pass

    def write_vectorstore(self, db: FAISS, path: str):
        pass 

    def similarity_search(self, db: FAISS, text: str, k: int):
        output = []
        for i in range(k):
            output.append(Document(page_content="foo", metadata={
                "Title": f"Title{i}",
                "Description": f"Description{i}",
                "Url": f"https://www.kaggle.com/c/{i}",
            }))
        return output

    def add_doc_to_vectorstore(self, db, docs):
        pass