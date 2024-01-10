import os

import openai
from openai_api_token import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY
# openai.api_key = os.environ['OPENAI_API_KEY']


def ask_llm(query: str) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": query},
        ]
    )

    result = ''
    for choice in response.choices:
        result += choice.message.content

    return result
