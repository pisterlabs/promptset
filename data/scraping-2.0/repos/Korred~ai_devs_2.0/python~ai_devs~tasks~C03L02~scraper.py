import os

import openai
from dotenv import load_dotenv
from icecream import ic
from utils.client import AIDevsClient
from tenacity import retry, wait_exponential
import requests


@retry(wait=wait_exponential())
def fetch_txt(url):
    ic("Fetching text...")
    # Set User-Agent header to avoid 403 error (bot detection)
    # https://www.whatismybrowser.com/guides/the-latest-user-agent/edge
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.2210.91"
    }

    r = requests.get(url, headers=headers)
    r.raise_for_status()
    return r.text


# Load environment variables from .env file
load_dotenv()

# Set OpenAI API key
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Get API key from environment variables
aidevs_api_key = os.environ.get("AIDEVS_API_KEY")

# Create a client instance
client = AIDevsClient(aidevs_api_key)

# Get a task
task = client.get_task("scraper")
ic(task.data)

# Get text URL
text_url = task.data["input"]

# Fetch text but retry if it fails
text = fetch_txt(text_url)


# Define system message
msg = task.data["msg"]

system_msg = f"""
{msg}

To answer the question, you can use the following text as context:
{text}
"""

# Define chat question
question = task.data["question"]

# Define chat completion
completion = openai.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": system_msg},
        {"role": "user", "content": question},
    ],
    max_tokens=400,
)

ic(completion)

answer = completion.choices[0].message.content

# Post an answer
response = client.post_answer(task, answer)
ic(response)
