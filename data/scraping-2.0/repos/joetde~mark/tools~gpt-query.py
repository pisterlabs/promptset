#!/usr/bin/env python

import openai
import os
import sys

openai.api_key = os.environ["OPENAI_API_KEY"]

prompt = sys.argv[1:]

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": " ".join(prompt)}]
)

print(response.choices[0].message.content.strip())

