import json
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
task = client.get_task("blogger")
ic(task.data)

# Get chapter topics and format user message
user_msg = "\n".join([f"{i+1}) {chapter}" for i, chapter in enumerate(task.data["blog"])])

# Create system message
system_msg = """
Act as a blogger and generate a blog post about pizza with chapters that will be provided as a list.
For each provided chapter, write 5-6 sentences that explain and describe the topic,
providing insightful information and specific proportions e.g. how much flour is needed to make a pizza. 

Return all the chapters as a JSON list of strings where every chapter is just one string.

Remember to write in Polish.
"""

# Get the completion
completion = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
    max_tokens=1000,
)
ic(completion)

# Get the chapters text (use json.loads to parse the JSON string returned in the content field)
chapters_text = json.loads(completion["choices"][0]["message"]["content"])

# Post an answer
response = client.post_answer(task, chapters_text)
ic(response)
