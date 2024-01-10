import os
import openai
openai.api_key = "OPENAI-API-KEY"

completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a poetic assistant, skilled in writing metal songs about research data"},
    {"role": "user", "content": "Compose a metal song that explains METS and MODS."}
  ]
)

print(completion.choices[0].message)
