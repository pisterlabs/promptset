# Load environment variables from .env
from dotenv import load_dotenv; load_dotenv()

# Fetch environment variables from .env
import os

# OpenAI Python SDK to allow us to make requests to GPT
import openai

# ===========================
# Welcome! This is the part of the file where you'll be writing your code.
# Feel free to add to it and experiment, but altering the starter code is not recommended.
# Make sure to read the comments carefully!
# Don't hesitate to ask for help if you're stuck. Happy coding!
# ===========================

# ===========================
# === MODIFY THIS SECTION ===
# ===========================

# Set OpenAI environment variables as shown in documenation
openai.api_version = '2023-05-15'
openai.api_type = os.getenv('OPENAI_API_TYPE')
openai.api_base = os.getenv('OPENAI_API_BASE')
openai.api_key = os.getenv('OPENAI_API_KEY')

# Call Azure to send the message to GPT and return the response string
def sendMessage(message: str) -> str:
    res = openai.ChatCompletion.create(
        engine='gpt-35-turbo',
        messages=[{'role': 'user', 'content': message}]
    )

    return res['choices'][0]['message']['content']
