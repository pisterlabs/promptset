#!/usr/bin/python3

import openai
import sys

query = sys.argv[1]

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "Translate given query to polish language"},
        {"role": "user","content": query }
    ]
)
print(response['choices'][0]['message']['content'])
