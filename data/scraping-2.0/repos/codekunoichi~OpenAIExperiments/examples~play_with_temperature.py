import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
  model="text-davinci-003",
  prompt="What is the meaning of life?",
  temperature=0.1,
  max_tokens=150,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0
)
print("Temperature = 0.1\n")
print(response)


response = openai.Completion.create(
  model="text-davinci-003",
  prompt="What is the meaning of life?",
  temperature=0.5,
  max_tokens=150,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0
)

print("Temperature = 0.5\n")
print(response)


response = openai.Completion.create(
  model="text-davinci-003",
  prompt="What is the meaning of life?",
  temperature=0.9,
  max_tokens=150,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0
)

print("Temperature = 0.9\n")
print(response)
