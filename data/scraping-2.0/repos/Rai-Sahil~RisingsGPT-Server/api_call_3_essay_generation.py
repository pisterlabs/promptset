# api_call_3_essay_generation.py

import os
import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set your OpenAI API key and model
openai.api_key = os.environ.get("OPEN_API")
gpt_model = "gpt-4"
# gpt_model = "gpt-3.5-turbo"

def gpt_response_3_essay_generation(messages):

    response = openai.ChatCompletion.create(
        model=gpt_model,
        temperature=0,
        max_tokens=3000,
        messages=messages  # Pass the updated messages to the API
    )

    # Print GPT-4's response
    print(f"\nGPT-4 Response:\n{response.choices[0].message.content}")

    # Return both the response and the updated messages for further use
    return response.choices[0].message.content
