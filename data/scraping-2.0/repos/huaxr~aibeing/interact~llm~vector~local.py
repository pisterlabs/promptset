# -*- coding: utf-8 -*-
# @Team: AIBeing
# @Author: huaxinrui@tal.com

# pip install chromadb
import os
import time
import uuid
from typing import List

from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from interact.llm.vector.base import Vector

class VectorDB(Vector):
    def __init__(self , collection, persist_directory=None):
        # todo: embedding 开元方案，下述不准确
        # self.embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.db = Chroma(collection_name= collection, embedding_function=OpenAIEmbeddings(), persist_directory=persist_directory)
        self.persist()
    def get_embeddings(self, texts: str) -> List[float]:
        return self.db._embedding_function.embed_documents([texts])[0]

    async def async_get_embeddings(self, texts: str) -> List[float]:
        res = await self.db._embedding_function.aembed_documents([texts])
        return res[0]

    def insert_text(self, texts: str):
        self.db.add_texts([texts])
        self.persist()
    def insert_text_with_unique(self, texts: str, unique: str):
        self.db.add_texts([texts], metadatas=[{"unique": unique}])
        self.persist()

    def search_by_str(self, query: str, k: int = 5):
        docs = self.db.similarity_search(query, k)
        return docs
    def search_by_str_with_unique(self, query: str, unique: str, k: int = 5):
        docs = self.db.similarity_search(query, k, filter={"unique": unique})
        return docs
    def insert_with_embedding(self, texts: str, embeddings:List[float],  user: str):
        self.db._collection.upsert(
            metadatas={"user": user}, embeddings=embeddings, documents=texts, ids=str(uuid.uuid1())
        )
        self.persist()

    def search_by_embedding(self, embedding: List[float], k:int = 3) -> List[str]:
        vector_res =  self.db.similarity_search_by_vector(embedding, k)

        content = []
        for doc in vector_res:
            page_content = doc.page_content.replace("\n", "")
            content.append(page_content)
        return content

    def count(self):
        return self.db._collection.count()

    def delete(self, contains_doc_str):
        self.db._collection.delete(where_document={"$contains": contains_doc_str})
        self.persist()

    def persist(self):
        self.db.persist()

def get_db_path() -> str:
    current_folder_path = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(current_folder_path, "chroma_db")
    print(db_path)
    return db_path

vdb = VectorDB(collection="AIBeing", persist_directory=get_db_path())
def init_db():
    with open("../../../data/sanguo-format.txt", "r", encoding="utf-8") as f:
        text = f.readlines()
        for i in text:
            print(i)
            vdb.insert_text(i.strip())

    print(vdb.count())
    print(vdb.search_by_str("刘备"))

def test():
    print(vdb.count())
    print(vdb.search_by_str("刘备"))

if __name__ == "__main__":
    current_timestamp = time.time()
    test()
    times = time.time() - current_timestamp
    print(times)

