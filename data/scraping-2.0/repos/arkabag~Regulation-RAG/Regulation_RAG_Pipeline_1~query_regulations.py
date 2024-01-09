from typing import Any, Dict, List, Optional
from llama_index import StorageContext, ServiceContext
from llama_index.indices.vector_store import VectorStoreIndex
from llama_index.vector_stores import PGVectorStore
import textwrap
from sqlalchemy import make_url
import psycopg2
import cohere
import openai 


class DocumentQueryEngine(object):
    def __init__(self):
        cohere_key = "your-cohere-api-key"  # Replace with your actual API key
        self.cohere_client = cohere.Client(cohere_key)
        # OpenAI initialization
        openai.api_key = "your-openai-api-key"  # Replace with your actual API key
        self.db_name = "regulation_embeddings"
        self.connection_string = "postgresql://testuser:testpwd@localhost:5433/regulation_embeddings"
        self.conn = psycopg2.connect(self.connection_string)
        self.conn.autocommit = True

        #Vector store setup
        url = make_url(self.connection_string)
        self.vector_store = PGVectorStore.from_params(
            database=self.db_name,
            host=url.host,
            password=url.password,
            port=url.port,
            user=url.username,
            table_name="new_york_regulation",
            embed_dim=1024
        )

        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        self.service_context = ServiceContext.from_defaults(chunk_size=1024)
        self.index = VectorStoreIndex.from_vector_store(vector_store=self.vector_store)
        self.query_engine = self.index.as_query_engine()


    def create_embedding_from_text(self, sentences: List[str]) -> List[List[float]]:
        embeddings = self.cohere_client.embed(sentences, model="embed-english-v3.0").embeddings
        return embeddings


    def answer(self, question):
        search_response = self.query_engine.query(question)
        context = ' '.join([chunk.text for chunk in search_response])  # Modify based on actual structure
        gpt_response = query_gpt4_with_context(context, question)
        return textwrap.fill(gpt_response, 100)

    
   # Reuse the GPT-4 query function from load_regulations.py
def query_gpt4_with_context(context, question):
    combined_prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    response = openai.Completion.create(
        engine="text-davinci-004",
        prompt=combined_prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()


if __name__ == '__main__':
    engine = DocumentQueryEngine()

    print(engine.answer("What population does this chapter apply to?"))
    print(engine.answer("What does the word 'city' mean in this chapter?"))
    print(engine.answer("What are the different classes in which 'multiple dwellings' are divided into?"))
