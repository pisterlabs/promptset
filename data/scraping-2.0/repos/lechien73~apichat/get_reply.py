import os

from openai import OpenAI

client = OpenAI(api_key = os.getenv("OPENAI_KEY"))

prompt = open("prompt.txt").read()

response = client.chat.completions.create(
    model="gpt-4-1106-preview",
    messages=[
        {'role': 'user', 'content': prompt}
    ],
    temperature=0,
    stream=True  # this time, we set stream=True
)

for chunk in response:
    print(chunk['choices'][0]['delta'])