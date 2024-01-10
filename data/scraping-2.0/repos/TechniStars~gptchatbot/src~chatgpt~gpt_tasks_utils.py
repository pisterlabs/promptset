import glob
import json
import os
import pickle
import time
from pathlib import Path

import openai

from src.config.config import CONFIG
from src.data.document import Document
from src.utils.chatgpt.token_check import get_token_count
from src.utils.timeout import timeout

openai.api_key = CONFIG['chatgpt_config']['api_key']


def trim_message_to_token_limit(message, system_prompt, token_limit=3500):
    token_count = get_token_count(system_prompt + message)
    while token_count > token_limit:
        message = message[:int(len(message) * 0.85)]
        token_count = get_token_count(system_prompt + message)
    return message


@timeout(40)
def generate_chat_response(message, system_prompt, functions=None, n_retry=3):

    for i in range(1, n_retry + 1):
        try:
            message = trim_message_to_token_limit(message, system_prompt)
            messages = []
            messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": message})

            if not functions:
                chat = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo", messages=messages
                )
                reply = chat.choices[0].message.content
            else:
                chat = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    functions=functions,
                    function_call={'name': "get_qa"}
                )
                reply = chat.choices[0].message.function_call.arguments
            return reply
        except Exception as e:
            print(f'ChatGPT failed ({e}). Trying again in 5 seconds... ({i}/{n_retry})')
            time.sleep(3)


def convert_json_str_to_dicts(response):
    try:
        # Parse the JSON string into a list of dictionaries
        response_json = json.loads(response)
        return response_json
    except json.JSONDecodeError as e:
        raise ValueError("Invalid JSON format: " + str(e))


def read_source_file(file_path):
    document = Document(file_path)
    return document.content


def load_remaining_file_paths(data_path, pickle_path):
    if not os.path.exists(pickle_path):
        files_left = glob.glob(str(Path(data_path, '*.*')))
    else:
        with open(pickle_path, 'rb') as file:
            files_left = pickle.load(file)
    return files_left
