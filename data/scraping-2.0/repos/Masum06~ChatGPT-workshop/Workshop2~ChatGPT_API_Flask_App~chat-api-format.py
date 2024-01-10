import os
import openai
openai.api_key = "OPENAI_API_KEY"

completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  max_tokens=100,
  temparature=0
)

print(completion.choices[0].message)

