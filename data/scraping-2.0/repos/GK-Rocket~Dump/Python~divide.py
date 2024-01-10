import os
import openai

openai.api_key = os.getenv("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")

response = openai.Completion.create(
  model="text-davinci-003",
  prompt="write a email to the boss about how he have been?",
  temperature=0.7,
  max_tokens=256,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)