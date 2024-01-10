import os
import openai

openai.api_key ="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"

response = openai.Completion.create(
  model="text-davinci-003",
  prompt="The following is a list of companies and the categories they fall into:\n\nApple, Facebook, Fedex\n\nApple\nCategory:",
  temperature=0,
  max_tokens=64,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0
)