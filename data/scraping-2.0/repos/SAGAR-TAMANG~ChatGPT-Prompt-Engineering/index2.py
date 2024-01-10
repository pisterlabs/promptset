# This Shows A Normal Chat Messages From ChatGPT

import openai
openai.api_key = "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"
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

def get_completion_from_messages(messages, model="gpt-3.5-turbo", temperature=0):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature, # this is the degree of randomness of the model's output
    )
#     print(str(response.choices[0].message))
    return response.choices[0].message["content"]