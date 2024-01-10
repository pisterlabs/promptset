import json

from camel.types import ModelType
from openai import OpenAI
import os
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def ask_gpt(msg):
    completion = client.chat.completions.create(
      model="gpt-4-1106-preview",
      messages=[
        {"role": "system", "content": "You are an document assistant, skilled in generate well formatted documents or codes from given conversation."},
        {"role": "user", "content": msg}
      ]

    )
    return completion.choices[0].message.content
