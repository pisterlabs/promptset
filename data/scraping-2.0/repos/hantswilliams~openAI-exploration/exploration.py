#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  5 19:07:57 2021

@author: hantswilliams


openAI exploration 

pip install openai



"""

import os
import openai
from dotenv import dotenv_values

os.chdir('/Users/hantswilliams/Documents/development/python_projects/openAI-exploration')
config = dotenv_values("./.env") 
openapikey = config["apikey"]


openai.api_key = config["apikey"]

response = openai.Completion.create(engine="davinci", prompt="This is a test", max_tokens=2)
print(response)




#### Parse unstructured data 
response = openai.Completion.create(
  engine="davinci",
  prompt="There are many fruits that were found on the recently discovered planet Goocrux. There are neoskizzles \
      that grow there, which are purple and taste like candy. There are also loheckles, which are a grayish blue fruit \
      and are very tart, a little bit like a lemon. Pounits are a bright green color and are more\
      savory than sweet. There are also plenty of loopnovas which are a neon pink flavor and taste \
      like cotton candy. Finally, there are fruits called glowls, which have a very sour and bitter \
      taste which is acidic and caustic, and a pale orange tinge to them.\n\nPlease make a table summarizing \
      the fruits from Goocrux\n| Fruit | Color | Flavor |\n| Neoskizzles | Purple | Sweet |\n| Loheckles | Grayish blue | Tart |\n",
  temperature=0,
  max_tokens=10,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0,
  stop=["\n\n"]
)

print(response)
