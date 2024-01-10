# Script for using gpt4 via openai api for normalisation of text
import os
import pandas as pd
from datetime import datetime
import openai
import config
import requests
import json
from dateutil.parser import parse
import re
import time

prompt = "Der folgende Text ist ein Brief aus dem 19. Jahrhundert in historischem Deutsch. Übertrage den Brief in eine moderne Ortographie, wo es notwendig ist. Verändere so wenig wie möglich: "

openai.api_key = config.openai_key

with open(os.path.join(os.getcwd(),'..','data','literature','Briefwechsel_pupikofer_lassberg.txt'), 'r', encoding='utf-8') as file:
    text = file.read()
letters = text.split('#')
new_letters = ""
for letter in letters:
    letter_to_be_send = re.sub(r'{.*?}', '', letter)
    content = prompt + letter_to_be_send
    messages = [{"role": "user", "content": content}]
    try:
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo-16k",messages=messages)
        normalized_letter = completion.choices[0].message.content
    except:
        time.sleep(3)
        try:
            completion = openai.ChatCompletion.create(model="gpt-3.5-turbo-16k",messages=messages)
            normalized_letter = completion.choices[0].message.content
        except:
            normalized_letter = "Error"
    letter = letter + f"\n[[{normalized_letter}]]"
    new_letters = new_letters + "\n#\n" + letter
    print(letter)
    time.sleep(1)

with open(os.path.join(os.getcwd(),'..','data','literature','pupikofer_normalized.txt'), 'w', encoding='utf-8') as file:
    file.write(new_letters)