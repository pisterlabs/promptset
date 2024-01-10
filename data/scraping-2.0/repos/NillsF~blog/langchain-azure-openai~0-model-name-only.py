import os
from langchain.embeddings import OpenAIEmbeddings
import dotenv

# Load environment variables from .env file
dotenv.load_dotenv()

# Create an instance of the OpenAIEmbeddings class using Azure OpenAI
embeddings = OpenAIEmbeddings(
    deployment=os.getenv("OPENAI_DEPLOYMENT_NAME"),
    chunk_size=1)

# Testing embeddings
txt = "This is how you configure it directly in the constructor."

# Embed a single document
e = embeddings.embed_query(txt)

print(len(e)) # should be 1536