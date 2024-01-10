'''
    File: gpt35.py
    Author: Mohamed
    Date: 01/04/2023
    @Last updated: 01/04/2023
    @By Mohamed
    For: SMS Chatbot
    Purpose: This is wrapper for chatgpt it takes msg from AT inbox and send it to chatgpt and return response in json
    License: GNU GPL v2.0 or Later versions
'''
import typing
import openai
import json

openai.api_key = 'YOUR-API-KEY'


def chatgpt35(msg: str) -> any:

    completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": msg}
    ]
    )
    return json.loads(json.dumps(completion.choices[0].message))['content']