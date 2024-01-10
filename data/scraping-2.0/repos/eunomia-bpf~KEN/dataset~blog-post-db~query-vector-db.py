from langchain.document_loaders import JSONLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
import os

# Define the file loader for JSON documents
loader = JSONLoader(
    file_path='./data/format-summary.json',
    jq_schema='.data[].content',
    json_lines=True
)

# Load the documents using the loader
documents = loader.load()

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings()

# Check if the vector database files exist
if not (os.path.exists("./data/vector_db.faiss") and os.path.exists("./data/vector_db.pkl")):
    # Create a new FAISS vector store and save it
    db = FAISS.from_documents(documents, embeddings)
    db.save_local("./data", index_name="vector_db")
else:
    # Load an existing FAISS vector store
    db = FAISS.load_local("./data", index_name="vector_db", embeddings=embeddings)

# Prompt the user for input
user_query = input("Enter your query: ")

# Perform a search in the vector store based on user query
results = db.search(user_query, search_type='similarity')
print(results)
