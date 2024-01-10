import openai
import os
import json

openai.api_key=os.getenv('OPENAI_API_KEY')
model = 'gpt-3.5-turbo'

def generate_chat(model,messages,temperature):
    response = openai.ChatCompletion.create(model=model,messages=messages,temperature=temperature)
    return response

messages = [
    {"role":"system","content":"you are helpful assitant"},
    {"role":"user","content":"Hello?"}
]

while True:
    try:
        temperature = int(input('온도를 입력하세요(0~2사이)?'))
        response = generate_chat(model,messages,temperature)
        answer = response['choices'][0]['message']['content']
        print(answer)
    except KeyboardInterrupt as e:#CTRL +C로 빠져나가기
        break