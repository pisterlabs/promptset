import os
import sys
import re
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, Qdrant
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.storage import (
    LocalFileStore,
)
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.schema import Document
from ai_utils import conversation_to_string
CACHED_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "cached_embedding")
from ai.ai_utils import DATE_FORMAT

class Retriever:
    def __init__(self, conversations, vector_db_type="chroma") -> None:
        fs = LocalFileStore(CACHED_PATH)
        underlying_embeddings = OpenAIEmbeddings()
        self.embedding = CacheBackedEmbeddings.from_bytes_store(
            underlying_embeddings, fs, namespace=underlying_embeddings.model
        )
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

        documents = []
        for conversation in conversations:
            text = conversation.get("information", [])
            text = "\n".join(text)

            if text:
                chunks = self.text_splitter.split_text(text)
                for chunk in chunks:
                    documents.append(Document(page_content=chunk))

        if not documents:
            self.db = Chroma(embedding_function=self.embedding)
            return

        if vector_db_type == "chroma":
        # Ensure that a recent version of SQLite3 is used
        # Chroma requires SQLite3 version >= 3.35.0
        # See https://gist.github.com/defulmere/8b9695e415a44271061cc8e272f3c300
            # if sys.platform == "linux":
            #     __import__("pysqlite3")
            #     sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
            self.db = Chroma.from_documents(documents, self.embedding)
        else:
            raise Exception("Not implemented")

    def query(self, query: str, top_k: int = 3):
        results = self.db.similarity_search(query, k=top_k)
        return [x.page_content for x in results]

    def add_text(self, text: str):
        self.db.add_text([text])

def build_retriever(conversations, vector_db_type="chroma"):
    return Retriever(conversations, vector_db_type)
