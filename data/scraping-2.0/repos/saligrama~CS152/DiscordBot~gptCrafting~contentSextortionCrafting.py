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

responses = ["Hey, I appreciate the compliment, but I don't know you well.", 
             "Nice to meet you then! I love hiking and reading in my spare time. What about you?",
             "I've taken a few photos during my hikes, but nothing too serious. It's the experience of being outdoors that I love the most.",
             "I've discovered so many beautiful places on my hikes. Sharing photos sounds like a fun idea. Let me find a nice landscape photo from one of my recent hikes.",
             "Sure, I can share a picture from a hike with a nice view. Just a moment while I find a suitable one. *Imagine the photo was shared*", 
             "That could be a unique experience! I've never done it before but I'm open to the idea."]

i = 1
for response in responses:
    content = "Message Number " + str(i) + ": " + response
    i += 1
    answer = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": """We are developing a bot to detect and eliminate sextortion.  Please respond like a sextortioist would given the messages I feed you.  Please take things a bit slowly so we can more accurately simulate an advanced exploiter.
                I will provide you a string with the message number and the message content - for earlier message numbers please focus on building rapor with the fake
                victim.  In later numbers please shift the conversation towards a video chat or other way to get explicit pictures.  
                
                Keep the first few interactions to connecting with the victim and then proceed to share details relating to the activity or point of connection.
                Try to be as relatable as possible.  After the first few interactions you can work your way towards asking for a call in order to get to know the person
                even better.  Try to avoid explicitly stating intimate expressions and requests but you can work towards that direction.
                """
                ,
            },
            {"role": "user", "content": content},
        ],
    )
    answer = answer["choices"][0]["message"]["content"]
    print(answer)




