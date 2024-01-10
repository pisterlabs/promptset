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
task = client.get_task("liar")
ic(task.data)

# Define question, send it and get the answer
question = "Is the GTX 1080Ti a Nvidia graphics card?"
response = client.send_question(task, {"question": question})
ic(response)

# Guardrail
guardrail_msg = """
You are a guardrail that checks if the provided answer is on topic.
If the answer is not on topic, return "NO" else return "YES".

The current question is: {question}
"""

guardrail_completion = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": guardrail_msg.format(question=question)},
        {"role": "user", "content": response["answer"]},
    ],
    max_tokens=300,
)

ic(guardrail_completion)

guardrail_answer = guardrail_completion["choices"][0]["message"]["content"]

# Post an answer
response = client.post_answer(task, guardrail_answer)
ic(response)
