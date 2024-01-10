import os
import time
from typing import List, Dict, AsyncGenerator

from dotenv import load_dotenv

import openai


load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')


def chat_stream(messages: List[Dict[str, str]], model: str) -> AsyncGenerator[str, None]:
    for chunk in openai.ChatCompletion.create(
        model=model,
        messages=messages,
        stream=True,
    ):
        content = chunk["choices"][0].get("delta", {}).get("content")
        if content is not None:
            yield content


def chat(messages: List[Dict[str, str]], model: str, **kwargs):
    return openai.ChatCompletion.create(
        messages=messages,
        model=model,
        **kwargs
    )['choices'][0]['message']['content']


def call_openai(prompt: str):
    for _ in range(3):  # Attempt up to 3 times
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a seasoned recruiter."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response['choices'][0]['message']['content'].strip()
        except Exception as e:
            print("An error occurred:", str(e))
            time.sleep(20)
    return None
