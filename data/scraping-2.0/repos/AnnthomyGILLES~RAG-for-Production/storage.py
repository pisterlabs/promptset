import os

from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv
from loguru import logger

import chromadb

load_dotenv()

EMBEDDING_MODEL = "text-embedding-ada-002"

embedding_function = OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API"), model_name=EMBEDDING_MODEL
)


class StoreResults:
    _instance = None

    def __init__(self):
        self.collection = self.chroma_client.get_or_create_collection(
            name="ray-documentation",
            embedding_function=embedding_function,
        )

    def __new__(cls):
        # Create instance if not already created
        if cls._instance is None:
            cls._instance = super(StoreResults, cls).__new__(cls)
            # Initialize ChromaDB client
            cls._instance.chroma_client = chromadb.PersistentClient(path="./chromadb")
        return cls._instance

    def __call__(self, batch):
        try:
            documents = batch["text"].tolist()
            embeddings = batch["embeddings"].tolist()
            ids = batch["index"].tolist()
            metadatas = [{"source": value} for value in batch["source"]]

            self.collection.add(
                embeddings=embeddings, documents=documents, metadatas=metadatas, ids=ids
            )
            logger.info(f"Successfully upserted {len(documents)} documents.")
        except Exception as e:
            logger.error(f"Failed to upsert documents: {e}")
        return {}


if __name__ == "__main__":
    store = StoreResults()
    results = store.collection.query(
        query_texts=["Tell me about Head Node"], n_results=2
    )
    print(results)
