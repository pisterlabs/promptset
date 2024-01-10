# from dotenv import load_dotenv
# import os
from abc import ABC, abstractmethod
# import pinecone
# import weaviate
# import chromadb
# from chromadb.config import Settings
# from langchain.embeddings.openai import OpenAIEmbeddings


class Ingestor(ABC):
    def __init__(self, file_path, embeddings):
        self.file_path = file_path
        self.embeddings = embeddings

    @abstractmethod
    def get_documents(self):
        pass

    @abstractmethod
    def ingest(self):
        pass
