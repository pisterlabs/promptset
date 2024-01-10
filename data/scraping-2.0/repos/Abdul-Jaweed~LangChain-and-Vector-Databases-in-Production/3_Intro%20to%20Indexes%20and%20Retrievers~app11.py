import os
from dotenv import load_dotenv

from langchain.embeddings import OpenAIEmbeddings
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


apikey = os.getenv("OPENAI_API_KEY")


# Define the documents

documents = [
    "The cat is on the mat.",
    "There is a cat on the mat.",
    "The dog is in the yard.",
    "There is a dog in the yard.",
]

# Initialize the OpenAIEmbeddings instance

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# Generate embeddings for the documents

document_embeddings = embeddings.embed_documents(documents)

# Perform a similarity search for a given query

query = "A cat is sitting on a mat."
query_embedding = embeddings.embed_query(query)

# Calculate similarity scores

similarity_scores = cosine_similarity([query_embedding], document_embeddings[0])

# Find the most similar document
most_similar_index = np.argmax(similarity_scores)
most_similar_document = documents[most_similar_index]

print(f"Most similar document to the query '{query}':")
print(most_similar_document)