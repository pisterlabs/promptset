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
task = client.get_task("inprompt")
ic(task.data)


# Parse the input data and create a dictionary to look up the text snippets by name
name_information = {}
for text in task.data["input"]:
    name = text.split(" ")[0]
    name_information[name] = text


# Find out the name of the person that the question is about
system_msg = "Based on the provided question, what is the name of the person that the question is about?"
question = task.data["question"]
name_completion = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": system_msg},
        {"role": "user", "content": question},
    ],
    max_tokens=100,
)

# Add information about the person/name to the system message
name = name_completion["choices"][0]["message"]["content"]
system_msg = f"Answer a question about the person using the following facts: {name_information[name]}"

# Ask a question about the person and get the answer
question_completion = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": name_information[name]},
        {"role": "user", "content": question},
    ],
    max_tokens=200,
)

answer = question_completion["choices"][0]["message"]["content"]

# Post an answer
response = client.post_answer(task, answer)
ic(response)
