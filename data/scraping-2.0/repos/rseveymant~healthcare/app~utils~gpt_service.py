import openai
import os
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

class GPTService:
    def __init__(self):
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def generate(self, prompt):
        response = openai.ChatCompletion.create(
            model="gpt-4",  # or whatever model you want to use
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response['choices'][0]['message']['content']
