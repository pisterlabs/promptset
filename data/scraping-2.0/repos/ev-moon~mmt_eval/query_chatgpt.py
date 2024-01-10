import os

import openai
import requests

from translation_utils import get_wmt_2014_translations

openai.api_key = os.getenv("OPENAI_API_KEY")
API_URL = "https://api.openai.com/v1/chat/completions"
ENGLISH = "en"
API_COOLDOWN = 20


def query_gpt(l1, l2, sentence):
    message = [
        {
            "role": "user",
            "content": f"Translate the following sentence from {l1} to {l2}: ```{sentence}```",
        }
    ]
    data = {"model": "gpt-3.5-turbo", "messages": message, "temperature": 0.7}
    response = requests.post(
        API_URL,
        json=data,
        headers={
            "Authorization": f"Bearer {openai.api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        return response.json()["choices"][0]["message"]["content"]
    except:
        print(response.json())
        raise RuntimeError("API exception")


get_wmt_2014_translations(API_COOLDOWN, query_gpt)