import os
import requests
import openai

# 設定 OpenAI API 的金鑰
OPENAI_API_KEY = "//Your open api key//"

# 初始化 OpenAI API
openai.api_key = OPENAI_API_KEY


def gpt35(role, q):
    """ 
    gpt-3.5-turbo 
    role: 設定角色
    q: 輸入的問題
    """

    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    temperature=0.7,
    messages=[
        {"role": "system", "content":role}, # 設定角色
        {"role": "user", "content": q}
    ])
    bot_response = response['choices'][0]['message']['content']
    return bot_response

def gpt4(role, q):
    """ 
    gpt-4
    role: 設定角色
    q: 輸入的問題
    """

    response = openai.ChatCompletion.create(
    model="gpt-4",
    temperature=0.7,
    messages=[
        {"role": "system", "content":role}, # 設定角色
        {"role": "user", "content": q}
    ])
    bot_response = response['choices'][0]['message']['content']
    return bot_response


