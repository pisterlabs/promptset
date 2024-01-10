import openai
from .vectore_store import VectorStore
import traceback

EMBEDDING_MODEL = "text-embedding-ada-002"

import openai

# Singleton class for handling vector encoding and storage
class VectorHandler:
    _instance = None

    def __new__(cls, openai_apikey=None, pinecone_apikey=None):
        if cls._instance is None:
            cls._instance = super(VectorHandler, cls).__new__(cls)
            openai.api_key = openai_apikey
            cls._instance.vector_store = VectorStore(pinecone_apikey)
            cls._instance.vector_store.initialize_pinecone()
        return cls._instance

    def encode_message(self, message):
        model = "text-embedding-ada-002"
        #print(f"Encoding message: {message}")
        response = openai.Embedding.create(input=[message], model=model)
        #print(f"response length: {len(response['data'][0]['embedding'])}")
        return response['data'][0]['embedding']

    def get_relevant_contexts(self, initial_message):
        print("Getting relevant contexts...")
        query_vector = self.encode_message(initial_message)
        return self.vector_store.fetch_context(query_vector)

    def store_additional_data(self, summary):
        #print(f"Storing additional data: {summary}")
        print("Storing additional data in vector store...")
        summary_vector = self.encode_message(summary)
        self.vector_store.store_additional_data(summary_vector, summary)

    async def close(self):
        await self.vector_store.close()

