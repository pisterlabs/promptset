import os
import openai

openai.api_key = os.getenv("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")

response = openai.Completion.create(
  model="text-davinci-003",
  prompt="Create an analogy for this phrase:\n\nQuestions are arrows in that:",
  temperature=0.5,
  max_tokens=60,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0
)

# Prompt
# Create an analogy for this phrase:

# Questions are arrows in that:
# Sample response
# Questions are like arrows in that they both have the power to pierce through the surface and uncover the truth that lies beneath.

