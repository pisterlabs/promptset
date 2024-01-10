import os
import openai
from dotenv import load_dotenv

load_dotenv()

# Load your API key from an environment variable or secret management service
openai.api_key = os.getenv("OPENAI_API_KEY")


def getComponents(text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are escrowGPT. Users give you the text from their escrow agreements"
                + "and you identify the names of the parties and the amount in escrow, based on the text."
                + "Your answers match this format: {'party1': '...', 'party2': '...', 'amount': '...'}",
            },
            {
                "role": "user",
                "content": text,
            },
        ],
    )
    print(response)
    return response["choices"][0]["message"]["content"]
