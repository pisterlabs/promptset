import pinecone
import os
import openai
from dotenv import load_dotenv
load_dotenv()

pinecone.init(api_key=os.getenv('PINECONE_API_KEY'), environment='gcp-starter')

client = openai.Client(api_key=os.getenv('OPENAI_API_KEY'))

def get_openai_embedding(text):
    response = client.embeddings.create(
        input=[text],
        model="text-embedding-ada-002"  # Choose the appropriate model
    )
    embedding = response.data[0].embedding
    return list(embedding)

index_name = 'theo-bot'

# Create or connect to an existing Pinecone index
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=2048)  # Dimension based on OpenAI's embedding size
index = pinecone.Index(index_name)

# Example text to be embedded
example_text = "Hello, this is a test text for Pinecone and OpenAI embeddings."

# Get the embedding using OpenAI
embedding = get_openai_embedding(example_text)

if not isinstance(embedding, list) or not all(isinstance(x, float) for x in embedding):
    raise ValueError("Embedding is not in the correct format")


# Upsert the vector with a unique ID
index.upsert(vectors={'example_text_id': embedding})

# Query to test
results = index.query(queries=[embedding], top_k=1)
print(results)