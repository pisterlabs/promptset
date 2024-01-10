#!/usr/bin/env python3

from time import sleep
from random import uniform
import os
import openai

openai.api_key = os.environ['GPT_KEY']

prompt = input("Enter your prompt \n -> ")

completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "user", "content": prompt},
  ]
)

response = completion.choices[0].message.content

for x in response:
    print(x, end='', flush=True)
    sleep(uniform(0, 0.1))
