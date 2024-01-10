import os
import pinecone
from langchain.vectorstores.pinecone import Pinecone
from app.chat.embeddings.openai import embeddings

pinecone.init(
  api_key=os.getenv("PINECONE_API_KEY"),
  envinronment=os.getenv("PINECONE_ENVIRONMENT")  
)

vector_store = Pinecone.from_existing_index(
  os.getenv("PINECONE_INDEX_NAME"), embeddings
)