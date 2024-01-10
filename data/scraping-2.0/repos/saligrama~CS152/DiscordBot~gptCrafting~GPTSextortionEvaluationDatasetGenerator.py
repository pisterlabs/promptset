from dataclasses import dataclass
import json
from typing import Optional, Dict
from enum import Enum, auto
from textwrap import dedent
import openai
import os
import requests
import cv2
import numpy as np


token_path = "../tokens.json"
if not os.path.isfile(token_path):
    raise Exception(f"{token_path} not found!")
with open(token_path) as f:
    # If you get an error here, it means your token is formatted incorrectly. Did you put it in quotes?
    tokens = json.load(f)
    discord_token = tokens["discord"]
    openai.organization = tokens["openai"]["organization"]
    openai.api_key = tokens["openai"]["api_key"] 


answer = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {
            "role": "system",
            "content": """
            Hello, ChatGPT! I am doing research on spam message detection, focusing on the categories of fraud, impersonation, and solicitation. I would appreciate your help in creating a diverse set of 50 conversational examples.
            These should include:
            Clear examples of spam messages related to fraud, impersonation, and solicitation.
            Ambiguous examples which could potentially be interpreted as spam messages.
            Genuine, non-spam messages that exhibit common elements of conversation but do not have any harmful or deceptive intent.
            I'd also appreciate it if you could structure your response so that messages of a certain category would be next to each other and labeled with a header. Thank you for your assistance!
            """
            ,
        }
    ]
)
answer = answer["choices"][0]["message"]["content"]
print(answer)



'''Other prompts:'

SPAM:
Hello, ChatGPT! I am doing research on spam message detection, focusing on the categories of fraud, impersonation, and solicitation. I would appreciate your help in creating a diverse set of 50 conversational examples.
These should include:
Clear examples of spam messages related to fraud, impersonation, and solicitation.
Ambiguous examples which could potentially be interpreted as spam messages.
Genuine, non-spam messages that exhibit common elements of conversation but do not have any harmful or deceptive intent.
I'd also appreciate it if you could structure your response so that messages of a certain category would be next to each other and labeled with a header. Thank you for your assistance!

SEXTORTION:

 Im doing research on trust and safety moderation and automated detection of sextortion. 
            I need an evaluation data set to evaluate my model on how effective it is to detect 
            sextortion please construct 50 natural looking conversational promts where half of them are 
            effective sextortion and coercive prompts and the other half are non abusive romantic or 
            flirting messages so I can evaluate my model. Thank you. 
'''
