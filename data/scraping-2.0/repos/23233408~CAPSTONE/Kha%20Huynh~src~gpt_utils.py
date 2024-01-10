import requests
import re
import openai

openai.api_key = "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"

def chat_with_chatgpt(prompt, test, model="gpt-3.5-turbo"):
    response = openai.Completion.create(
        engine=model,
        prompt=prompt.replace("{test}",test),
        max_tokens=250,
        n=1,
        stop=None,
        temperature=0.0,
    )

    message = response.choices[0].text.strip()
    return message


# For local streaming, the websockets are hosted without ssl - http://
HOST = 'localhost:5000'
URI = f'http://{HOST}/api/v1/generate'

# For reverse-proxied streaming, the remote will likely host with ssl - https://
# URI = 'https://your-uri-here.trycloudflare.com/api/v1/generate'


def run(prompt,test):
    request = {
        'prompt': prompt.replace("{test}",test),
        'max_new_tokens': 250,
        'temperature': 0.0,
        'top_p': 0.1,
        'length_penalty': 5,
        'early_stopping': True,
        'seed': 0,
    }

    response = requests.post(URI, json=request)
    # print(response.json()['results'][0]['text'])

    return response.json()['results'][0]['text']