import openai
import os

conversation = []
openai.api_key = os.getenv('OPENAI_KEY') 

user = input("Ask something to chatgpt : ")
conversation.append({"role": "user", "content": user})
response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=conversation)
answer = response["choices"][0]['message']['content']
print(answer)