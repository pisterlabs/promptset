#!/usr/bin/python3
#
# Keywords.py
# Christopher Minson
#
#

import os
import string
import json
import datetime
import openai


PATH_TEXT = "./data/text/"
PROMPT = "Extract keywords from this text:\n\n"



openai.api_key = os.getenv("OPENAI_API_KEY")

talk_list_text = os.listdir(PATH_TEXT)
for file_name in talk_list_text:
    path_text = PATH_TEXT + file_name
    f =  open(path_text)
    content = f.read()
    f.close()
    content = content[0:10000]
    content = content.replace('\n', '')
    #print(content)
    prompt_text = f'Extract keywords from this text:\n\n {content}'
    prompt_text = prompt_text.replace('\'','')
    prompt_text = prompt_text.replace('"','')
    #prompt_text = f"Extract keywords from this text:\n\n see the dog run after the cat"
    print(prompt_text)
    

    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=prompt_text,
        temperature=0.5,
        max_tokens=600,
        top_p=1.0,
        frequency_penalty=0.8,
        presence_penalty=0.0
    )
    print(response)

    keywords = response["choices"][0]["text"]
    list_keywords = keywords.split('\n')
    #list_keywords = [k for k in list_keywords if len(k) > 2]
    list_keywords = [k for k in list_keywords if len(k.lstrip('-')) > 2] 
    print(list_keywords)
    exit()


