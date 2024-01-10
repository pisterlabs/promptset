#!/usr/bin/env python

# pip install openai
import openai

# APIキーの設定
openai.api_key = "your_openai_apikey"

msg = input("You->")

messages = []
while(True):

    messages.append({"role": "user", "content": msg})
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
    )
    res = response.choices[0]["message"]["content"]
    
    messages.append({"role": "assistant", "content": res})

    print("chatGPT->",res.strip())
    msg = input("You->")


