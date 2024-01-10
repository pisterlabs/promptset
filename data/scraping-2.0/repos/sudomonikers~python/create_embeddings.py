import openai
import os 

openai.api_key = os.getenv('OPENAI_API_KEY')

embedding = openai.Embedding.create(
    input="Your text goes here", model="text-embedding-ada-002"
)
print(len(embedding.data[0].embedding))
