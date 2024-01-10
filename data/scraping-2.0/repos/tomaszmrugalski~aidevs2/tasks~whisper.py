# This is a solution to the whisper task from aidevs2 course
# See https://zadania.aidevs.pl/ for details
#
# Bibliography:
# - https://platform.openai.com/docs/guides/speech-to-text
#
# Necessary dependecies: openai
# Installation: pip install openai
#
# The overall goal is to:
# 1. Get a token from aidevs
# 2. Get the task from aidevs
# 3. Download the audio file if it doesn't exist locally.
# 4. Use whisper-1 model to transcript the audio file
# 5. Send the answer to aidevs
# 6. Profit!

import requests
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
TASK = 'whisper'

# STEP 1: Get the token from aidevs
url = BASE_URL + '/token/' + TASK
print(f'aidevs: Getting {url}, sending {APIKEY}')
page = requests.post(url, json={ "apikey": APIKEY })
data = json.loads(page.text)
token = data['token']
print(f"aidevs: My token is {token}")

# STEP 2: Get the task from aidevs
url = BASE_URL + '/task/' + token

query={  }
print(f"aidevs: Sending {query} to {url}")
page2 = requests.post(url, data=query)
data2 = json.loads(page2.text)

print(f"aidevs: the task is {data2}")

# SOLUTION
file = 'https://zadania.aidevs.pl/data/mateusz.mp3'
file_local = 'data/mateusz.mp3'

# STEP 3 - first need to download the file (if not present locally)
if not os.path.exists(file_local):
    print(f'Downloading {file} to {file_local}')
    os.makedirs(os.path.dirname(file_local), exist_ok=True)
    page = requests.get(file)
    with open(file_local, 'wb') as f:
        f.write(page.content)
else:
    print(f"File {file_local} already exists, skipping download")

# STEP 4: Use whisper-1 model to transcript the audio file
openai.api_key = OPENAI_KEY
audio_file = open(file_local, "rb")
transcript = openai.audio.transcriptions.create(model="whisper-1", file=audio_file)
print(transcript.text)

# STEP 5: Send the answer
url = BASE_URL + '/answer/' + token
answer={ 'answer': transcript.text }
print(f'aidevs: Getting {url}, sending {answer}')
page3 = requests.post(url, json = answer)
data3 = json.loads(page3.text)
print(f"aidevs: /answer/token returned {data3}")
