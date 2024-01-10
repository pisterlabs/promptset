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

text = [
  "The girl with the black and white puppies have a ball.", # The girl has a ball
  "Yolanda has her notebook.", # ok
  "Its going to be a long day. Does the car need it's oil changed?", # Homonyms
  "Their goes my freedom. There going to bring they're suitcases.", # Homonyms
  "Your going to need your're notebook.", # Homonyms
  "That medicine effects my ability to sleep. Have your heard of the butterfly affect?",
  "This phrase is to cherck chatGPT for speling abilitty" # spelling
]

for t in text:
	prompt = f"Proofread and correct: ```{t}```"
	response = get_completion(prompt)
	print (response)
