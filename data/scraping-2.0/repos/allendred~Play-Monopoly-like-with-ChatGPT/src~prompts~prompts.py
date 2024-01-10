'''
Author: allendred
Date: 2023-05-20 17:51:00
LastEditors: allendred
LastEditTime: 2023-05-23 09:59:50
FilePath: /Play-Monopoly-like-with-ChatGPT/src/prompts/prompts.py
Description: Follow your heart
'''
import os
import openai
from dotenv import load_dotenv

load_dotenv(verbose=True)

# Load your API key from an environment variable or secret management service
openai.api_key = os.getenv('OPENAI_API_KEY')
print(openai.api_key)
def answer(text):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": text}],
        timeout=1000
    )
    return completion.choices[0]['message']['content']

def ask(prefix='',prompt_template='''Q: What is the meaning of life?''',history=[]):
    text = prefix +  '\n' + prompt_template
    result = answer(text)
    return result