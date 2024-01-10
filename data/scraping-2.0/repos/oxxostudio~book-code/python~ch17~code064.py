# Copyright © https://steam.oxxostudio.tw

import openai
openai.api_key = '你的 API Key'

messages = []
while True:
    msg = input('me > ')
    messages.append({"role":"user","content":msg})   # 添加 user 回應
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        max_tokens=128,
        temperature=0.5,
        messages=messages
    )
    ai_msg = response.choices[0].message.content.replace('\n','')
    messages.append({"role":"assistant","content":ai_msg})   # 添加 ChatGPT 回應
    print(f'ai > {ai_msg}')


