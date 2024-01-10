from .vector import VectorRepository
from langchain.vectorstores import Pinecone
import pinecone
from os import environ
from langchain.embeddings.base import Embeddings


class PineconeRepository(VectorRepository):
    def __init__(self, embedding: Embeddings):
        super().__init__(embedding)
        api_key = environ["PINECONE_API_KEY"]
        env = environ["PINECONE_ENV"]
        index = environ["PINECONE_INDEX"]

        # Initialize Pinecone
        pinecone.init(
            api_key=api_key,
            environment=env,
        )

        # Create an index and a vector store
        self.index = pinecone.Index(index)
        self.vectorstore = Pinecone(self.index, embedding=embedding, text_key="text")

    def ingest(self, product):
        """Ingest product into vector store"""
        # Convert product to (id, vector, metadata)
        vector = [
            (product.id, values, {"text": str(product)})
            for values in self.embedding.embed_documents([str(product)])
        ]
        self.index.upsert(vectors=vector)
