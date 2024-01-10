import os
from dotenv import load_dotenv
import openai

# Set model and batch size parameters
EMBEDDING_MODEL = "text-embedding-ada-002"
QUERY_MODEL = "gpt-4"


def setup_openai():
    load_dotenv()

    # Initialize OpenAI API
    openai.api_key = os.getenv("OPENAI_API_KEY")
