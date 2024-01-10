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
task = client.get_task("functions")
ic(task.data)

# Define a function specification
function = {
    "name": "addUser",
    "description": "Adds a new user",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "User name",
            },
            "surname": {
                "type": "string",
                "description": "User surname",
            },
            "year": {
                "type": "integer",
                "description": "User birth year",
            },
        },
    },
    "required": ["name", "surname", "year"],
}


# Post an answer
response = client.post_answer(task, function)
ic(response)
