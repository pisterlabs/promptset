from openai import OpenAI
from dotenv import load_dotenv
import os
import json

load_dotenv()

LLM_DATA_FEED = os.environ['LLM_DATA_FEED']
LLM_DATA_REPLY = os.environ['LLM_DATA_REPLY']

def main():
  client = OpenAI()

  with open(LLM_DATA_FEED, 'r') as file:
      data = json.load(file)

  response = client.chat.completions.create(
    model="gpt-4",
    messages=data
  )

  response_text = response.choices[0].message.content
  print(response_text)

  with open(LLM_DATA_REPLY, 'w') as file:
    file.write(response_text)

main()
