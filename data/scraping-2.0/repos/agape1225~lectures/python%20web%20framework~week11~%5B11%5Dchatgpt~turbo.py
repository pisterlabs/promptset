import os
import openai

api_key = os.environ["api_key"]
openai.api_key = api_key 

Q = "chatGPT에 대해서 설명해줘"

messages = []
messages.append({"role": "user", "content":Q})
completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",  messages=messages)

res= completion.choices[0].message['content']
print(res)