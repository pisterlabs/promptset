import openai
import config

OPENAI_API_KEY = config.api_key
openai.api_key = OPENAI_API_KEY

msg = input("Enter your value: ")
messages=[{"role": "user", "content": msg}]

response = openai.ChatCompletion.create(
model="gpt-4",
max_tokens=500,
temperature=1.2,
messages = messages)

print(response.choices[0].message.content)