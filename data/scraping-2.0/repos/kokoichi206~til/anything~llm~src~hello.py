import openai

import os
# from dotenv import load_dotenv

# load_dotenv()

# response = openai.ChatCompletion.create(
#     model="gpt-3.5-turbo",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant!"},
#         {"role": "user", "content": "Hello, I'm John!"},
#     ],
# )

# print(response)

# stream で実装。
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant!"},
        {"role": "user", "content": "Hello, I'm John!"},
    ],
    stream=True,
)

for chunk in response:
    # <class 'openai.openai_object.OpenAIObject'>
    # print(type(chunk))
    choice = chunk["choices"][0]
    if choice["finish_reason"] is None:
        print(choice["delta"]["content"])

