import os
import openai
import time

from config import Config

openai.api_key = Config.OPENAI_API_KEY

# messages=[
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": "Knock knock."},
#     {"role": "assistant", "content": "Who's there?"},
#     {"role": "user", "content": "Orange."},
# ]

def gpt3_all(prompt, max_tokens=3500, temperature=0.0):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    return response

def gpt3_text(prompt, max_tokens=3500, temperature=0.0):
    return gpt3_all(prompt, max_tokens=max_tokens, temperature=temperature).choices[0].text


def gpt3_text_prompt_all(prompt, text_chunks, max_tokens=3500, temperature=0.0):
    responses = []
    for i, chunk in enumerate(text_chunks):
        gpt_prompt = f"{prompt}{chunk}"
        print("Getting response for chunk: ", i)
        gpt_response = gpt3_text(gpt_prompt, max_tokens, temperature)
        responses.append(gpt_response)
    return responses


def gpt35_all(messages, temperature = 0.0, max_tokens=None):
    retry_count = 10
    for i in range(0,retry_count):
        while True:
            try:
                response = openai.ChatCompletion.create(
                    model = "gpt-3.5-turbo",
                    messages = messages,
                    temperature = temperature,
                    max_tokens = max_tokens,
                )
                return response
            except Exception as e:
                print(f"API Error: {e}")
                print(f"Retrying {i+1} time(s) in 10 seconds...")
                time.sleep(10)
                continue
            break

def gpt35_text(messages, temperature = 0.0, max_tokens=None):
    return gpt35_all(messages, temperature).choices[0]['message']['content']

def gpt35_text_stream(messages, temperature = 0.0, max_tokens=None):
    retry_count = 10
    for i in range(0,retry_count):
        while True:
            try:
                response = openai.ChatCompletion.create(
                    model = "gpt-3.5-turbo",
                    messages = messages,
                    temperature = temperature,
                    max_tokens = max_tokens,
                    stream = True,
                )
                return response
            except Exception as e:
                print(f"API Error: {e}")
                print(f"Retrying {i+1} time(s) in 10 seconds...")
                time.sleep(10)
                continue
            break
