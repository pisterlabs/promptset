import os
import openai



openai.api_key = "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"

response = openai.Completion.create(
  model="text-davinci-003",
  prompt="You: What have you been up to?\nFriend: Watching old movies.\nYou: Did you watch anything interesting?\nFriend:",
  temperature=0.5,
  max_tokens=60,
  top_p=1.0,
  frequency_penalty=0.5,
  presence_penalty=0.0,
  stop=["You:"]
)
print(response)