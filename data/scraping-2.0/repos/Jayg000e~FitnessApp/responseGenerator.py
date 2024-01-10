import os
from dotenv import load_dotenv

import json

# Load environment variables from the .env file
load_dotenv()

# Retrieve the API key from the environment variable
api_key = os.getenv("OPENAI_API_KEY")

# Set the API key for the OpenAI library
os.environ["OPENAI_API_KEY"] = api_key

# Now you can use the OpenAI API
import openai

def responseGenerator(records):

    # Define a conversation
    conversation = [
        {"role": "system", "content": "You are an expert in fitness training"},
        {"role": "user", "content": "base on my fitness record"+str(records)+"give me some advise for future training"},
    ]

    # Send the conversation to the OpenAI API for completion
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Choose an appropriate model
        messages=conversation,
    )

    # Get the assistant's reply from the API response
    # assistant_reply = response['choices'][0]['message']['content']
    assistant_reply = response['choices'][0]['message']['content']

    return assistant_reply