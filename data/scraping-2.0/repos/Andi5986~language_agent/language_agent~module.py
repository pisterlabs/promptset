import os
from dotenv import load_dotenv
import openai

# Load environment variables from .env file
load_dotenv()

def call_openai_api(prompt):
    openai.api_key = os.getenv("OPENAI_API_KEY")

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    return response['choices'][0]['message']['content']
