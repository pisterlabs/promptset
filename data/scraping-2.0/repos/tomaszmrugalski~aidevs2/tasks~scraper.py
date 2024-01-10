# This is a solution to the scraper task from aidevs2 course
# See https://zadania.aidevs.pl/ for details
#
#
# The overall goal is to:
# 1. Get a token from aidevs
# 2. Get the task from aidevs, that includes a link to a document and a question about it.
# 3. Retrieve the source text.
# 4. Ask GPT to answer question about the source text. Provide source text in a system message.
# 5. Send the answer to aidevs
# 6. Profit!

import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder
import json
import yaml
import os.path
import os
import openai

def load_apikey():
    """Loads the aidevs API key from ~/.aidevs2"""
    with open(os.path.expanduser('~/.aidevs2'), 'r') as file:
        api_key = yaml.safe_load(file)
    return api_key['APIKEY']

def load_openai_key():
    """Loads the openai API key from ~/.aidevs2"""
    with open(os.path.expanduser('~/.aidevs2'), 'r') as file:
        api_key = yaml.safe_load(file)
    return api_key['OPENAI_KEY']

BASE_URL = 'https://zadania.aidevs.pl'
APIKEY = load_apikey()
OPENAI_KEY = load_openai_key()
TASK = 'scraper'

# STEP 1: Get the token from aidevs
url = BASE_URL + '/token/' + TASK
print(f'aidevs: Getting {url}, sending {APIKEY}')
response1 = requests.post(url, json={ "apikey": APIKEY })
data = json.loads(response1.text)
token = data['token']
print(f"aidevs: My token is {token}")

# STEP 2: Get the task from aidevs
url = BASE_URL + '/task/' + token
query={  }
print(f"aidevs: Sending {query} to {url}")
response2 = requests.post(url, data=query)
data2 = json.loads(response2.text)
link = data2['input']
question = data2['question']

print(f"aidevs: the task is {data2}")

# STEP 3: SOLUTION:
# Retrieve the source text.
url = 'https://api.openai.com/v1/chat/completions'
headers = { 'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/119.0' }
response2 = requests.get(link, headers=headers)
print(response2.text)

# Use retrieved text as context in the system message.
system_prompt = f'Odpowiedz po polsku na pytania zwiazane z tym kontekstem. Odpowiadaj zwięźle. Ogranicz długość odpowiedzi do 200 znaków:\n\n {response2.text}'
body = { "messages": [{ "role": "system", "content": system_prompt}, { "role": "user", "content": question}], "model": "gpt-3.5-turbo"}
headers = { 'Content-Type': 'application/json', 'Authorization': f'Bearer {OPENAI_KEY}' }
print(f'OpenAI: Getting {url}, using {OPENAI_KEY}')
page = requests.post(url, json=body, headers=headers)
data = json.loads(page.text)
print(f"OpenAI: Response body: {data}")
answer  = { 'answer': data['choices'][0]['message']['content'] }

# STEP 4: Send the answer to aidevs
url = BASE_URL + '/answer/' + token
print(f'aidevs: Sending answer {answer} to {url}.')
response3 = requests.post(url, json = answer)
data3 = json.loads(response3.text)
print(f"aidevs: /answer/token returned {data3}")
