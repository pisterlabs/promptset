import os
import openai

# Function to get vector embedding from ADA-2
def get_embedding_from_ada2(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0]['embedding']

text = "Text to embed goes here"
embedding = get_embedding_from_ada2(text)

print("Embedding:", embedding)