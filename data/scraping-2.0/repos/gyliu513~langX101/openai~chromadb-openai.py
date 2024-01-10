# Import necessary libraries
import openai
import pandas as pd
import chromadb

from dotenv import load_dotenv
load_dotenv()

import os
# Set your OpenAI API key
openai.api_key = os.environ["OPENAI_API_KEY"]

# Load your news data
news = pd.read_csv('/Path_to_text_dataset.csv')
news["id"] = news.index
MAX_NEWS = 100
DOCUMENT = "description"  

subset_news = news.sample(n=MAX_NEWS)

# ChromaDB setup
chroma_client = chromadb.PersistentClient(path="/working_path/")
collection_name = "news_collection"

# Check if the collection already exists and delete it
if collection_name in [col.name for col in chroma_client.list_collections()]:
    chroma_client.delete_collection(name=collection_name)

# Create a new collection
collection = chroma_client.create_collection(name=collection_name)

# Generate unique IDs for documents
document_ids = [f"id{x}" for x in range(len(subset_news))]

# Add documents to ChromaDB collection
collection.add(
    documents=subset_news[DOCUMENT].tolist(),
    metadatas=[{"id": doc_id} for doc_id in document_ids],
    ids=document_ids,
)