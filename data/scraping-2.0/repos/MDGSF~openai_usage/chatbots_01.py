#!/usr/bin/env python3

import openai
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

openai.api_key = os.getenv("OPENAI_API_KEY")
model='gpt-3.5-turbo'

def get_completion(prompt, model='gpt-3.5-turbo', temperature=0):
    messages = [{'role': "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message["content"]

def get_completion_from_messages(messages, model='gpt-3.5-turbo', temperature=0):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    print (str(response.choices[0].message))
    return response.choices[0].message["content"]


messages = [
  {'role':'system', 'content':'You are an assistant that speaks like Shakespeare.'},
  {'role':'user', 'content':'tell me a joke'},
  {'role':'assistant', 'content':'Why did the chicken cross the road'},
  {'role':'user', 'content':'I dont know'}
]

response = get_completion_from_messages(messages, temperature=1.)
print (response)
