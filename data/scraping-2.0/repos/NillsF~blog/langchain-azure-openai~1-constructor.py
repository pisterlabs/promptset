from langchain.embeddings import OpenAIEmbeddings
import dotenv
import os

# Load environment variables from .env file
dotenv.load_dotenv()

# Create an instance of the OpenAIEmbeddings class using Azure OpenAI
embeddings = OpenAIEmbeddings(
    openai_api_type="azure",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_API_BASE"),
    deployment=os.getenv("OPENAI_DEPLOYMENT_NAME"),
    model=os.getenv("OPENAI_MODEL_NAME"),
    chunk_size=1)

# Testing embeddings
text = "This is how you configure it directly in the constructor."

# Embed a single document
e = embeddings.embed_query(text)

print(len(e)) # should be 1536