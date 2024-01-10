import os
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.redis import Redis

# Load environment variables from .env file
load_dotenv()

# Update the file path to the correct relative path
loader = TextLoader("data/thefourcorners.txt")  # Assuming the script is in ./src
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
docs = text_splitter.split_documents(documents)

# Use OpenAI as the embeddings provider
embeddings = OpenAIEmbeddings()
rds = Redis.from_documents(
    docs, embeddings, redis_url="redis://localhost:6379", index_name="chunk"
)
