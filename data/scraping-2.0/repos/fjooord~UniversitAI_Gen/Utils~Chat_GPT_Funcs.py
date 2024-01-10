import re
import os
import json
import openai
from time import time,sleep
import textwrap
from io import BytesIO
import requests
import re
import sys
import backoff  # for exponential backoff
        
def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()


def save_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)
        
def save_file2(filepath, content):
    with open(filepath, 'a', encoding='utf-8') as outfile:
        outfile.write(content)


openai.api_key = 'Enter Key'

def chat_gpt(prompt, engine='gpt-4', temp=0.25, top_p=1.0, tokens=3000, freq_pen=0.0, pres_pen=0.0, stop=['asdfasdf', 'asdasdf'], role=None):
    max_retry = 5
    retry = 0
    prompt = prompt.encode(encoding='ASCII',errors='ignore').decode()
    if role is None:
        role = "You are an experienced children's book writer with over 20 years of expertise in creating best-selling picture books for children aged 2-8"
    else:
        role = role.encode(encoding='ASCII',errors='ignore').decode()
    while True:
        try:
            return gpt_with_backoff(prompt, engine, temp, top_p, tokens, freq_pen, pres_pen, stop, role)
            
            """openai.ChatCompletion.create(
              model=engine,
              messages=[
                {"role": "system", "content": role},
                {"role": "user", "content": prompt}
              ],
              temperature=temp,
              top_p=top_p,
              max_tokens=tokens,
              frequency_penalty=freq_pen,
              presence_penalty=pres_pen,
            )"""

            text = completion.choices[0].message['content']
            usage = completion.usage
            return text, usage
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            sleep(1)

@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def gpt_with_backoff(prompt, engine='gpt-4', temp=0.25, top_p=1.0, tokens=3000, freq_pen=0.0, pres_pen=0.0, stop=['asdfasdf', 'asdasdf'], role=None):
    prompt = prompt.encode(encoding='ASCII',errors='ignore').decode()
    if role is None:
        role = "You are an experienced children's book writer with over 20 years of expertise in creating best-selling picture books for children aged 2-8"
    else:
        role = role.encode(encoding='ASCII',errors='ignore').decode()
    while True:
        completion = openai.ChatCompletion.create(
            model=engine,
            messages=[
            {"role": "system", "content": role},
            {"role": "user", "content": prompt}
            ],
            temperature=temp,
            top_p=top_p,
            max_tokens=tokens,
            frequency_penalty=freq_pen,
            presence_penalty=pres_pen,
        )

        text = completion.choices[0].message['content']
        usage = completion.usage
        return text, usage
        
def generalized_gpt_prompt(path, tag_values, engine = 'gpt-4', tokens = 1000, temp=0.7, index = -1, role=None):
    """
    This function takes in the path to the original prompt file
    and the current tags with their corresponding values

    We then iterate through all the tags and replace the tag in the prompt with the value
        - Any tag that exist will get replaced, but those that don't exist will do nothing 
        - This may increase runtime slightly, but reduces the number of functions that will be needed

    We pass the prompt to chat gpt and take the return value

    Note: The post processing may vary for each call
    """
    prompt = open_file(path)
  
    for tag, value in tag_values.items():
        
        try:
            prompt = prompt.replace(tag, value)
        except:
            if index != -1:
                try:
                    prompt = prompt.replace(tag, value[index])
                except:
                    continue

    return chat_gpt(prompt, temp=temp, engine=engine, tokens = tokens, role=role)

