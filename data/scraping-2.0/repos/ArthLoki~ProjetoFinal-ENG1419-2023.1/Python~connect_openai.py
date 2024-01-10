import openai
from dotenv import load_dotenv
import os

load_dotenv()

secret_key = os.getenv("SECRET_KEY_OPENAI")

openai.api_key = secret_key

def call_gpt(prompt):
    completions = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[ {"role": "user", "content": prompt }],
    )

    print(completions.choices[0].message.content)

    return completions.choices[0].message.content
