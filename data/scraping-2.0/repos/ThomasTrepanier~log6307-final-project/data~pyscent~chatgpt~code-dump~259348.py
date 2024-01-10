from abc import ABC, abstractmethod
import redis
import openai
from src.semantic_code.index.document.base_entity import VectorizableCodeEntity

class EmbeddingModel(ABC):
    @abstractmethod
    def create_embedding(self, text: str):
        pass

class OpenAIEmbeddingModel(EmbeddingModel):
    def create_embedding(self, text: str):
        # Use OpenAI API to create embedding
        # Please replace with actual OpenAI API call
        return openai.Embedding.create(text)

class EmbeddingStorage(ABC):
    @abstractmethod
    def store(self, key: str, vector):
        pass

    @abstractmethod
    def retrieve(self, key: str):
        pass

class RedisEmbeddingStorage(EmbeddingStorage):
    def __init__(self, host='localhost', port=6379, db=0):
        self.r = redis.Redis(host=host, port=port, db=db)

    def store(self, key: str, vector):
        self.r.set(key, vector)

    def retrieve(self, key: str):
        return self.r.get(key)

class FunctionEntity(VectorizableCodeEntity):
    def __init__(self, name: str, docstring: str, signature: str, embedding_model: EmbeddingModel, embedding_storage: EmbeddingStorage):
        super().__init__(docstring)
        self.name = name
        self.signature = signature
        self.embedding_model = embedding_model
        self.embedding_storage = embedding_storage

    def to_vector(self):
        """
        Convert the entity to a vector representation. In this example, it's a dictionary representation.
        """
        # Create a natural language representation of the entity
        natural_language_representation = f"{self.name} {self.signature} {self.docstring}"

        # Use the embedding model to create an embedding
        embedding = self.embedding_model.create_embedding(natural_language_representation)

        # Store the embedding in the embedding storage
        self.embedding_storage.store(self.name, embedding)

        return embedding
