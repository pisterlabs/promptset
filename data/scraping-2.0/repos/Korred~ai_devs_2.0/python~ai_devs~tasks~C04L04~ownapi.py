import os

import openai
from dotenv import load_dotenv
from icecream import ic
from utils.client import AIDevsClient

# Load environment variables from .env file
load_dotenv()

# Set OpenAI API key
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Get API key from environment variables
aidevs_api_key = os.environ.get("AIDEVS_API_KEY")

# Create a client instance
client = AIDevsClient(aidevs_api_key)

# Get a task
task = client.get_task("ownapi")
ic(task.data)

# Get API URL from environment variables
api_url = os.environ.get("API_URL")
assistant_endpoint = f"{api_url}/assistant"

response = client.post_answer(task, assistant_endpoint)
ic(response)
