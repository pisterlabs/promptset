import chromadb

# Initialize Chroma DB client
client = chromadb.PersistentClient(path="./db")
collection = client.get_collection(name="my_collection")
from langchain.embeddings import HuggingFaceEmbeddings

# Initialize GPT4All embeddings
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

# Get user input
query = input("Enter your query: ")

# Convert query to vector representation
query_vector = embeddings.embed_query(query)

# Query Chroma DB with the vector representation
results = collection.query(query_embeddings=query_vector, n_results=2 , include=["documents"])

# Print results
for result in results["documents"]:
    for i in result:
        print(i)