from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import chromadb

# create embeddings here
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Creates a chromadb persistent directory if it doesn't exist and at the same time creates a persistent client that can be used to interact withh the persistent db.
# If a persistent directory has already been created before, only the client is returned.
import os
workingDirectory = os.getcwd()
dbDirectory = os.path.join(workingDirectory, "ChromaDB/db")
persistent_client = chromadb.PersistentClient(path=dbDirectory)