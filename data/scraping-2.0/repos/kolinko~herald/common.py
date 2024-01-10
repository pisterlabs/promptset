# various helper functions for interacting with OpenAI, HN, cache, etc

import requests
import json
import random
import hashlib
import time
import datetime

import tiktoken
encoding = tiktoken.get_encoding("cl100k_base") # gpt2 for gpt3, and cl100k_base for gpt3turbo

from openai import OpenAI
try:
    from api_keys import organisation, api_key
except:
    print("You need to setup api keys first.\nEdit api_keys.py.default.py, adding your API keys, and rename the file to api_keys.py")
    exit()

openai = OpenAI(organization=organisation, api_key=api_key  )


#openai.organization = organisation
#openai.api_key = api_key

import redis
r = redis.Redis(host='localhost', port=6379, db=0)

try:
    r.ping()
except redis.exceptions.ConnectionError as ex:
    print("Redis server not enabled. Error: ", str(ex))
    print("Please install or start Redis server.")
    exit()

def md5(s):
    return hashlib.md5(s.encode('utf-8')).hexdigest()

def ai16k(system, prompt, retry=True, cache=True):
    return ai(system, prompt, "gpt-3.5-turbo-16k", retry=retry, cache=cache)


def ai3(system, prompt, retry=True, cache=True):
    return ai(system, prompt, "gpt-3.5-turbo", retry=retry, cache=cache)

def ai(system, prompt, json=False, model="gpt-4-1106-preview", retry=True, cache=True):
    cache_key = f'ai-cache:{model}:' + md5(system+'***'+prompt)
    if cache and r.exists(cache_key):
        return r.get(cache_key).decode('utf-8')

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt}
    ]

    while True:
        try:
            completion = openai.chat.completions.create(model=model, messages=messages, response_format={'type':'json_object' if json else 'text'})
            result = completion.choices[0].message.content
            r.set(cache_key, result)
            return result

        except Exception as e:
#            if not retry:
#                raise e
            # Print the error message in red color
            print("\033[91m" + f"Error occurred: {str(e)}" + "\033[0m")
            time.sleep(1)

        print('WTF')

def count_tokens(s):
    input_ids = encoding.encode(s)
    return len(input_ids)

def in_cache(url):
    return r.exists(url)

def download(url):
    response = requests.get(url)
    response.raise_for_status() # Check for any HTTP errors
    return response.content

def download_and_cache(url, cache_only=False, key_prefix=''):
    # Check if the content is already cached
    if r.exists(key_prefix+url):
        res = r.get(key_prefix+url)
        if res is not None:
            return res
    elif cache_only:
        return None

    while True:
        try:
            # If not cached, download and cache the content
            response = requests.get(url)
            response.raise_for_status() # Check for any HTTP errors
            content = response.content
            r.set(key_prefix+url, content)
            return content

        except Exception as e:
            # Print the error message in red color
            print("\033[91m" + f"Error occurred: {str(e)}" + "\033[0m")
            # Sleep for some time before retrying
            time.sleep(random.randint(5, 60))

def json_fetch(kind, id, cache_only=False):
    url = f"https://hacker-news.firebaseio.com/v0/{kind}/{id}.json?print=pretty"

    result = download_and_cache(url, cache_only)
    if cache_only and result is None:
            return None

    return json.loads(result)

def pretty_time(ts):
    dt = datetime.datetime.fromtimestamp(ts)
    return dt.strftime("%Y-%m-%d %H:%M:%S")

# basic profiling functions
start_time = time.time() 
def reset():
    global start_time
    start_time = time.time() 

def elapsed():
    elapsed_time = (time.time() - start_time) * 1000 # Get the elapsed time in milliseconds
    print(f"Time elapsed: {elapsed_time:.2f} ms")

