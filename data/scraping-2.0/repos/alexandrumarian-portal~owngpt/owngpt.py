#!/usr/bin/env python3


import openai
import os

os.environ['OPENAI_API_KEY'] = 'TOKEN GOES HERE'
openai.api_key = os.getenv('OPENAI_API_KEY')

##or
#import getpass
#key = getpass.getpass('Paste your key:')
#openai.api_key = key


prompt = input("Please enter input:\n")
messages = [
    {'role': 'system', 'content': 'Answer as concisely as possible.'},
   #{'role': 'system', 'content': 'Answer as detailed as possible.'},
    {'role': 'user', 'content': prompt},
]
#roles => system, user, assistant

response = openai.ChatCompletion.create(
    model = 'gpt-3.5-turbo',
    messages = messages,
    temperature = 0.5, # values between 0 - 2 ; default 1
    top_p = 0.1, # 1=use all words in dictionary; 0.1 = use 10% of words in dictionary
    max_tokens = 1000

)
#print(response) ###prints whole output
#print(response['choices'][0]['message']['content']) ###alternative 
print('--Response--\n')
print(response['choices'][0].message.content)
