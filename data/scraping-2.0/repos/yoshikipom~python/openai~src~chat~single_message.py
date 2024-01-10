from openai import OpenAI
import os

client = OpenAI()

content = 'hello'
chat_completion = client.chat.completions.create(
    model='gpt-4',
    messages=[{"role": "user", "content": content}]
)

print(chat_completion.choices[0].message.content)
