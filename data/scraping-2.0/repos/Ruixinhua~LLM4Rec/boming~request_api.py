import json
import urllib.request
import os
from openai import OpenAI

def request_from_llama2(user_content):
    model_name = "meta-llama/Llama-2-70b-chat-hf"

    url = "http://bendstar.com:8000/v1/chat/completions"
    req_header = {
        'Content-Type': 'application/json',
    }

    chat_completion = json.dumps({
        "model": model_name,
        "messages": [{"role": "system", 
                      "content": 'You serve as a personalized news recommendation system'}, 

                      {"role": "user", "content": user_content}],
        "temperature": 0,
    })

    req = urllib.request.Request(url, data=chat_completion.encode(), method='POST', headers=req_header)

    with urllib.request.urlopen(req) as response:
        body = json.loads(response.read())

    return body['choices'][0]['message']['content']

def request_from_gpt(user_content, model='gpt-4-1106-preview'):
    model_name = model

    client = OpenAI()

    chat_completion = client.chat.completions.create (
        model=model_name,
        messages=[{"role": "system", "content": 'You serve as a personalized news recommendation system'}, 
                    {"role": "user", "content": user_content}],
        temperature=0,
    )

    return chat_completion.choices[0].message.content
    