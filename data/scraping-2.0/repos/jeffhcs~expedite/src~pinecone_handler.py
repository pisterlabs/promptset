import os
from langchain.llms import OpenAI

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
#pinecone
import pinecone
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_API_ENV = "asia-southeast1-gcp-free"
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

llm = OpenAI()
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)
index = pinecone.Index("ifrs17")

#embeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.document_loaders import TextLoader
from langchain.document_loaders import TextLoader

def embed(text):
    embedding_model = OpenAIEmbeddings()
    vector = embedding_model.embed_query(text)
    return (vector)

def pinecone_upsert(step_description, step_history):
    vector = embed(step_description)
    return index.upsert([(step_history, vector)])

def pinecone_query(query):
    vector = embed(query)
    dict = index.query(vector=vector, top_k =1, include_values=True, include_metadata=True)
    return dict.matches[0].id


if __name__ == "__main__":

    example_step = "example_step"
    step_history = "example_history"

    pinecone_upsert(example_step, step_history)
    response = pinecone_query("example_step")
    print(response)