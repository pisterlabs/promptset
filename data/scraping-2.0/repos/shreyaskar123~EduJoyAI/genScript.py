import os
import openai
import re
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.environ.get('OPENAI_KEY')


def generate_storyline(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=prompt,
    )

    storyline_text = response["choices"][0]

    return storyline_text
