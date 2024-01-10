import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "Tu es un assistant intelligent"},
    {"role": "user", "content": "Explique simplement ce que tu es"},
    ],
    temperature=0,
)

print(completion['choices'][0]['message']['content'])