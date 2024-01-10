from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings

from typing import List


class ChromaDB():
    def __init__(self,persist_dir):
        self.persist_dir = persist_dir
        self.db = None

    def save_embeddings(self,docs:List,embeddings):
 
        self.db = Chroma.from_documents(
            documents=docs, embedding= embeddings, persist_directory=self.persist_dir
        )
        print(self.db._collection)
        print(self.db.__str__)
        self.embedding = embeddings

    def presist_db(self):
        self.db.persist()

    def load_db(self, embedding_function):
        self.db = Chroma(persist_directory=self.persist_dir, embedding_function=embedding_function)


## example
#chroma = ChromaDB()
#embeddings = OpenAIEmbeddings()
#MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
#embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
#chroma.save_embeddings(['test'],embeddings,"./tubegptdb")
#chroma.presist_db()
