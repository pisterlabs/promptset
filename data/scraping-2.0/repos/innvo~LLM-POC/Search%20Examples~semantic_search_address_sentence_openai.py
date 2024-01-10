import openai
from langchain.embeddings.openai import OpenAIEmbeddings
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Clear the terminal
os.system('cls' if os.name == 'nt' else 'clear')

## Set local environment variables
OPENAI_API_KEY=os.getenv("OPEN_API_KEY")

## Too Many Deimsions
embeddings = OpenAIEmbeddings()

# Define a function to get the GPT-3 completion for a string
def get_embedding(address):
    print("in search_embedding")
    address_embedding = embeddings.embed_query(address)
    return address_embedding


# Addresses to Compare
address1 = 'ABC Apple St New York NY'
address2 = 'ABC Apple Street New York New York'

# Get the GPT-3 completion for each address
address1_embedding= get_embedding(address1)
address2_embedding= get_embedding(address2)

print(address1_embedding)  

# Calculate the cosine similarity
similarity = 1 - cosine_similarity([address1_embedding], [address2_embedding])
print("Semantic Similarity between " + address1 + " and " + address2 + " is: " + str(similarity[0][0]))
