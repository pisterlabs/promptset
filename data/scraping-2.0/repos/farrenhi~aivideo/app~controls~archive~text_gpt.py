# grab content for generative AI tool
# input: keyword or a specific topic
# output: a statement
# limit: do 10 sec trial firstly!
import requests
from dotenv import load_dotenv
import os
load_dotenv()  # take environment variables from .env.

# from models.model import *
import openai
import os

openai.api_key = os.getenv('chatgpt_api_key')
# messages = [ {"role": "system", "content":  
#               "You are a intelligent assistant."} ] 

# while True: 
#     message = input("User : ") 
#     if message: 
#         messages.append( 
#             {"role": "user", "content": message}, 
#         ) 
#         chat = openai.ChatCompletion.create( 
#             model="gpt-3.5-turbo", messages=messages 
#         ) 
#     reply = chat.choices[0].message.content 
#     print(f"ChatGPT: {reply}") 
#     messages.append({"role": "assistant", "content": reply}) 

import requests

# Define the API endpoint and parameters
api_url = 'https://api.openai.com/v1/chat/completions'
api_key = os.getenv('chatgpt_api_key')  # Replace with your actual API key
headers = {
    'Authorization': f'Bearer {api_key}',
    'Content-Type': 'application/json'
}

# Define the prompt and other options
prompt = 'Once upon a time'
max_tokens = 50

# Make the API request
response = requests.post(
    api_url,
    headers=headers,
    json={
        'model': 'gpt-3.5-turbo',
        'messages': [{'role':'system', 'content':'You are a helpful assistant.'}, {'role':'user', 'content': prompt}],
        'max_tokens': max_tokens
    }
)

# Print the response
print(response.json())
