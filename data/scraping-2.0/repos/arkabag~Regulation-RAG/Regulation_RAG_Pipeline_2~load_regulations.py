from typing import Any, Dict, List, Optional
from PostgresLocalConnector import PostgresLocalConnector
from pathlib import Path
from llama_index import download_loader

from llama_index import StorageContext, ServiceContext
from llama_index.indices.vector_store import VectorStoreIndex
from llama_index.vector_stores import PGVectorStore
import textwrap
from sqlalchemy import make_url
import psycopg2
from pdf_parser import PdfParser
import cohere # New import for Cohere SDK
import numpy as np  # New import for NumPy
import openai
import sys

# Cohere client initialization
cohere_key = "tL1xX7zfTh3ZdvO99FQB3wops91s7xn92Cy5vxln"  # Replace with your actual API key
co = cohere.Client(cohere_key)

openai.api_key = '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'  # Replace with your actual OpenAI API key



class DocumentIngestor(object):
    def __init__(self, path):
        self.postgres_helper = PostgresLocalConnector()
     #   self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2') #not using this embedding model anymore

        StringIterableReader = download_loader("StringIterableReader")
        loader = StringIterableReader()
        pdf_parser = PdfParser(path)
        paragraphs = pdf_parser.get_paragraphs()
        self.documents = loader.load_data(texts=paragraphs)

    def create_embedding_from_text(self, sentences: List[str]) -> List[List[float]]:
        #Using Cohere for embeddings
        try:
            embeddings = co.embed(sentences, input_type="search_document", model="embed-english-v3.0").embeddings
            return np.asarray(embeddings).tolist()
        except Exception as e:
            raise Exception(f"An error occurred while creating embeddings: {e}")
            
        

# Function to query GPT-4 with enriched context
def query_gpt4_with_context(context, question):
    combined_prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    try:
        response = openai.Completion.create(
            engine="text-davinci-004",  # Use appropriate GPT-4 model
            prompt=combined_prompt,
            max_tokens=150  # Adjust as needed
        )
    except openai.error.OpenAIError as e:
        print(f"An error occurred while querying the GPT-4 model: {e}")
        sys.exit(1)
        
    


if __name__ == '__main__':
    file = './documents/MultipleDwellingLaw.pdf'
    db_name = "regulation_embeddings"
    table_name = "new_york_regulation"
    embed_model = "embed-english-v3.0"

    generator = DocumentIngestor(file)

    try:
        connection_string = f"postgresql://testuser:testpwd@localhost:5433/{db_name}"
        conn = psycopg2.connect(connection_string)
        conn.autocommit = True
    except psycopg2.OperationalError as e:
        print(f"Failed to connect to the database: {e}")
        sys.exit(1)

    try:
        with conn.cursor() as c:
            c.execute(f"TRUNCATE TABLE new_york_regulation")
            print(f"Table new_york_regulation truncated to accommodate new embeddings.")
    except psycopg2.errors.UndefinedTable:
        print(f"Table new_york_regulation was not found. It will be generated fresh.")
    except Exception as e:
        print(f"An error occurred while truncating the table: {e}")
        sys.exit(1)

    url = make_url(connection_string)
    vector_store = PGVectorStore.from_params(
        database=db_name,
        host=url.host,
        password=url.password,
        port=url.port,
        user=url.username,
        table_name=table_name,
        embed_dim=1024
    )


    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    service_context = ServiceContext.from_defaults(chunk_size=1024, embed_model=embed_model)
    

    index = VectorStoreIndex.from_documents(
        generator.documents,
        storage_context=storage_context,
        service_context=service_context,
        show_progress=True
    )

    query_engine = index.as_query_engine()

    # Perform the preliminary similarity search
    question = "What is the name of this document?"

    # Formulate context from search response
    # Modify this part as needed based on the actual structure of 'search_response'
    context = ' '.join([chunk.text for chunk in search_response])  # Assuming each chunk has a 'text' attribute
    # Query GPT-4 with the context and question
    gpt_response = query_gpt4_with_context(context, question)
    print(f"Question: {question}\nAnswer: {textwrap.fill(gpt_response, 100)}")


