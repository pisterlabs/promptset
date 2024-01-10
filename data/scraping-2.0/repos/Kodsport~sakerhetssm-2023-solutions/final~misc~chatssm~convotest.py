import openai
from secrets import OPENAI_KEY

openai.api_key = OPENAI_KEY

messages = []

while True:
    messages.append({"role": "user", "content": input()})
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
    messages.append(response["choices"][0]["message"])
    print(response["choices"][0]["message"]["content"])

# [
#         # {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": input()},
#         # {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
#         # {"role": "user", "content": "Where was it played?"}
#     ]
