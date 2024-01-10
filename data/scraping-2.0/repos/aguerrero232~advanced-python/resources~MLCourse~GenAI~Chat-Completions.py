import os
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")

prompt = input("Enter your prompt: ")

completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
  ]
)

print("\n" + completion.choices[0].message.content)

print("\nComplete response message:\n")
print(completion)

prompt = input("\nEnter a string for sentiment analysis: ")

completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "Classify user messages as positive or negative sentiment."},
    {"role": "user", "content": prompt}
  ]
)

print("\n" + completion.choices[0].message.content)

print("\nComplete response message:\n")
print(completion)

print("\nTell me a joke (with low temperature)")
completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  temperature = 0.2,
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Tell me a joke."}
  ]
)

print("\n" + completion.choices[0].message.content)

