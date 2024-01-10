# This code is for v1 of the openai package: pypi.org/project/openai
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import os


_ = load_dotenv(find_dotenv())

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {
            "role": "system",
            "content": "You will be provided with a sentence in English, and your task is to translate it into Hindi."
         },
        {
            "role": "user",
            "content": "This is a Flower"
        }
    ],
    temperature=0,
    max_tokens=256
)

print(response)