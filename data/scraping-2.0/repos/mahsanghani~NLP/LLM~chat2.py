import os
import openai

openai.api_key = os.getenv("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")

response = openai.Completion.create(
  model="text-davinci-003",
  prompt="The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.\n\nHuman: Hello, who are you?\nAI: I am an AI created by OpenAI. How can I help you today?\nHuman: I'd like to cancel my subscription.\nAI:",
  temperature=0.9,
  max_tokens=150,
  top_p=1,
  frequency_penalty=0.0,
  presence_penalty=0.6,
  stop=[" Human:", " AI:"]
)

# Prompt
# The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.

# Human: Hello, who are you?
# AI: I am an AI created by OpenAI. How can I help you today?
# Human: I'd like to cancel my subscription.
# AI:
# Sample response
# I understand, I can help you with canceling your subscription. Please provide me with your account details so that I can begin processing the cancellation.