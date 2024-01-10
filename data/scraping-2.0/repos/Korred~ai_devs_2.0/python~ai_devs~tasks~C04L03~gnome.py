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
task = client.get_task("gnome")
ic(task.data)

# Define the system
system_msg = """
Your task is to analyze a provided image. The image may or may not contain a gnome.
If it does contain a gnome, you should return the color of the gnomes hat in polish (e.g. czerwona, niebieska etc.).
If it does not contain a gnome, just return 'ERROR', nothing else.
"""

gnome_analyzer = openai.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[
        {"role": "system", "content": system_msg},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": task.data["url"]},
            ],
        },
    ],
)

# Extract the color of the gnome hat
answer = gnome_analyzer.choices[0].message.content
ic(answer)

# Post answer
response = client.post_answer(task, answer)
ic(response)
