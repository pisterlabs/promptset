import os
import openai

openai.api_key = os.getenv("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")

response = openai.Completion.create(
  model="text-davinci-003",
  prompt="Write a restaurant review based on these notes:\n\nName: The Blue Wharf\nLobster great, noisy, service polite, prices good.\n\nReview:",
  temperature=0.5,
  max_tokens=64,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0
)

# Prompt
# Write a restaurant review based on these notes:

# Name: The Blue Wharf
# Lobster great, noisy, service polite, prices good.

# Review:
# Sample response
# I recently visited The Blue Wharf and had a wonderful experience. The lobster was excellent and the prices were very good. The only downside was the noise level, but the polite service more than made up for it. Highly recommend!