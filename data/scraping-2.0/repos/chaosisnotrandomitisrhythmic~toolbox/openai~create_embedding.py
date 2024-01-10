import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

MODEL = "text-embedding-ada-002"

response = openai.Embedding.create(input="Your text string goes here", model=MODEL)
embeddings = response["data"][0]["embedding"]
print(len(embeddings))
