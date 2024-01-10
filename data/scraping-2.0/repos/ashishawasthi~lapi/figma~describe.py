import json
import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

with open('summary.json', 'r') as infile:
    figma_summary_json = json.load(infile)

chat_messages = [
    {"role": "system", "content": "Give a short summary of Figma document e.g. purpose and actions"},
    {"role": "user", "content": f"```json\n{figma_summary_json}\n```"}
]

response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=chat_messages,
)

print(response['choices'][0]['message']['content'])
