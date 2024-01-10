import os
import openai

openai.api_key = "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"

response = openai.Completion.create(
  model="text-davinci-003",
  prompt="how to fine tuning our own model",
  temperature=0,
  max_tokens=1000,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)

print(response)
