import os
import openai

openai.api_key = os.getenv("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")

response = openai.Completion.create(
  model="text-davinci-003",
  prompt="Extract the name and mailing address from this email:\n\nDear Kelly,\n\nIt was great to talk to you at the seminar. I thought Jane's talk was quite good.\n\nThank you for the book. Here's my address 2111 Ash Lane, Crestview CA 92002\n\nBest,\n\nMaya\n\nName:",
  temperature=0,
  max_tokens=64,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0
)

# Prompt
# Extract the name and mailing address from this email:

# Dear Kelly,

# It was great to talk to you at the seminar. I thought Jane's talk was quite good.

# Thank you for the book. Here's my address 2111 Ash Lane, Crestview CA 92002

# Best,

# Maya

# Name:
# Sample response
# Maya
# Mailing Address: 2111 Ash Lane, Crestview CA 92002