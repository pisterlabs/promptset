import openai


import os
import openai

openai.api_key = ""
openai.api_base = "http://localhost:5000"

completion = openai.ChatCompletion.create(
    model="codellama",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say: Hello World!"},
    ],
    stream=True,
)

r = ""
for chunk in completion:
    # r += str(chunk.choices[0].delta.content)
    print(chunk.choices[0].delta.content, flush=True, end="")
# print(r)
