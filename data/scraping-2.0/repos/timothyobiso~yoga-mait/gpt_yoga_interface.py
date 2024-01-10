# Author: Ben Soli
"""
This module contains the functions used for interacting with chat gpt
"""

import openai
import os
from query_classifier import classify
import pandas as pd
import numpy as np

# read in api key stored in a txt file
with open('openai_api_key.txt', 'r', encoding='utf8') as key:
    openai.api_key = key.readlines()[0]

model = 'gpt-3.5-turbo'

# queries chat gpt and returns the output as a string
def ask_chat_gpt(prompt: str) -> str:
    response = openai.ChatCompletion.create(
            model=model, 
            messages=[{'role': 'user', 'content': prompt}],
            temperature=.1,
            max_tokens=1024)
    return response['choices'][0]['message']['content']

# takes in user query and turns into gpt prompt
def query_to_prompt(query: str) -> str:
    if 'pose' not in query or 'poses' not in query:
        query = 'pose for ' + query
    if 'yoga' not in query:
        query = 'yoga ' + query
    return f'list of {query} no descrptions only americanized names'


if __name__ == '__main__':

    ask_chat_gpt('list of yoga poses good for balance no descriptions')





