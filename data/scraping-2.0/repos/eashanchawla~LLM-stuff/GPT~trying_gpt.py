import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")
print(openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[{'role': 'system', 'content': 'You are a helpful math tutor'}, {'role': 'user', 'content': 'Can you explain the primes theorem to me? '}],
  max_tokens=17,
  temperature=0
))