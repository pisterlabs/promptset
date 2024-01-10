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
task = client.get_task("moderation")
ic(task.data)

# Check text snippets via OpenAI moderation API
result_list = []
for text in task.data["input"]:
    moderation_response = openai.Moderation.create(
        input=text,
    )

    ic(moderation_response)

    flagged = moderation_response["results"][0]["flagged"]
    result_list.append(int(flagged))

# Post an answer
response = client.post_answer(task, result_list)
ic(response)
