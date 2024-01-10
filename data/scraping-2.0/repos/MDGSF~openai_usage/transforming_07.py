#!/usr/bin/env python3

import openai
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

openai.api_key = os.getenv("OPENAI_API_KEY")
model='gpt-3.5-turbo'

def get_completion(prompt, model='gpt-3.5-turbo'):
    messages = [{'role': "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message["content"]


data_json = { "resturant employees": [
  {"name": "Shyam", "email": "shyamjaiswal@gmail.com"},
  {"name": "Bob", "email": "bob32@gmail.com"},
  {"name": "Jai", "email": "jai87@gmail.com"}
]}

prompt = f"""
Translate the following python dictionary from JSON to HTML \
table with column headers and title: {data_json}
"""

response = get_completion(prompt)
print (response)
