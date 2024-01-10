import openai
from time import time,sleep
import re


with open('openaiapikey.txt', 'r') as infile:
    open_ai_api_key = infile.read()
openai.api_key = open_ai_api_key


def gpt3_completion(prompt, engine='text-davinci-001', temp=1.0, top_p=1.0, tokens=200, freq_pen=0.5, pres_pen=0.5, stop=['<<END>>']):
    try:
        response = openai.Completion.create(
            engine=engine,
            prompt=prompt,
            temperature=temp,
            max_tokens=tokens,
            top_p=top_p,
            frequency_penalty=freq_pen,
            presence_penalty=pres_pen,
            stop=stop)
        text = response['choices'][0]['text'].strip()
        text = re.sub('\s+', ' ', text)
        return text
    except Exception as oops:
        print('Error:', oops)
        return None