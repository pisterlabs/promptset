import os
import openai

openai.api_key = "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"

completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are an assistant."},
    {"role": "user", "content": "How are you doing today?"}
  ]
)



print(completion.choices[0].message['content'])