import openai
from openai import OpenAI

from pathlib import Path

import os
from dotenv import load_dotenv
load_dotenv()

open_ai_key = os.getenv("OPEN_AI_KEY")

# set API KEY
client = OpenAI(api_key = open_ai_key)

def get_response(messages, model="gpt-4-1106-preview"):
    response = client.chat.completions.create(
        model=model,
        messages = messages,
        max_tokens=None,
        temperature=0.9,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.6,
    )

    return response.choices[0].message.content