import os
import openai
import time
import json

import pandas as pd
from preprocess.augment.augment import count_token, count_tokens
from tenacity import retry, wait_random_exponential, stop_after_attempt

import os
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY


@retry(wait=wait_random_exponential(multiplier=1, max=10), stop=stop_after_attempt(3))
def get_response(text):
    start = time.time()
    tokens = count_tokens(text)

    response = openai.ChatCompletion.create(
        model=(ver := "gpt-3.5-turbo" if tokens < 4000 else "gpt-3.5-turbo-16k"),
        messages=text,
        temperature=0
    )
    price = 0.0015 * tokens * 1.3
    price += 0.002 * count_token(response['choices'][0]['message']['content']) * 1.3
    return response['choices'][0]['message']['content'], price, time.time() - start, ver


def convert_prompt(document):
    content = f"""
    You will be provided with a description of welfare services. 
    Perform the following actions:

    1 - Write a two sentence of description of who might need this service.
    2 - Transform the description into a question of someone describing the situation in conversational tone.
    3 - Translate the conversation to Korean.

    Output the final result in json format.

    Final format:
    {{
        "description" : <description> 
        "conversation" : <conversation>
        "translation" : <translation>
    }},

    Service description:
    {document}

    Situation should be related to the service description.
    """
    return [{"role": "user", "content": content}]


if __name__ == "__main__":
    data = {}
    for i, filename in enumerate(sorted(os.listdir('data/articles'))):
        if i % 3 != 0:
            continue
        with open(f"data/articles/{filename}") as f:
            document = f.read()
        res, price, delay, ver = get_response(convert_prompt(document))

        try:
            data[i] = {
                "data": json.loads(res),
                "desc": {"filename": filename,
                         "title": document.split('<span style="font-weight: bold">')[1].split('</span>')[0].strip(),
                         "index": i,
                         "category": filename.split("_")[0],
                         "version": ver,
                         }
            }
        except json.decoder.JSONDecodeError:
            print(res)
            continue

        with open("preprocess/augment_qa/rec_data.json", "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print("Data: \n", res)
        print(f"Price: {price:.2f} Time: {delay:.2f} Version: {ver}")
        print()
