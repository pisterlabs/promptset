# script to test the ada call for embeddings

import openai
import sys
sys.path.append('/Users/busterblackledge/')
from keys import openai_API_key

openai.api_key = openai_API_key

content = "hello world"
print(f"String for embedding: {content}")

content = content.encode(encoding='ASCII',errors='ignore').decode()
response = openai.Embedding.create(input=content, engine='text-embedding-ada-002')
embedding = response['data'][0]['embedding']

print(f"Embedded string from ada: {embedding[1]} - which is of type {type(embedding)}")


