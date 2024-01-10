import os
import sys
import glob
import json
import openai
import pickle
import getpass
from datetime import datetime

def load_chat_history(file_path):
    try:
        with open(file_path, 'rb') as f:
            chat_history = pickle.load(f)
    except FileNotFoundError:
        chat_history = []

    return chat_history

def save_chat_history(file_path, chat_history):
    with open(file_path, 'wb') as f:
        pickle.dump(chat_history, f)

if len(sys.argv) < 2:
    print("Usage: geist.py <openai_api_path> <user_input>")
    sys.exit(1)

openai.api_key_path = sys.argv[1]

chatml_path = os.path.dirname(os.path.abspath(__file__))

chatml_files = glob.glob(chatml_path + "/*.chatml")

chat_history_path = 'geist.pkl'

chatml = []

for chatml_file in chatml_files:
    with open(chatml_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            message = json.loads(line)
            current_timestamp = datetime.utcnow().isoformat().split('.')[0] + "Z"
            common_fields = {
                'role': message['role'],
                'content': message['content']
            }
            if 'name' in message:
                common_fields['name'] = message['name']
            chatml.append(common_fields)

chat_history = chatml + load_chat_history(chat_history_path)

user_input = sys.argv[2]
whoiam = getpass.getuser()
current_timestamp = datetime.utcnow().isoformat().split('.')[0] + "Z "
prompt = {"role": "user", "content": current_timestamp + user_input, "name": whoiam}

completion = openai.ChatCompletion.create(
    model="gpt-4",
    #model = "gpt-3.5-turbo",
    messages = chat_history + [prompt],
)

chat_history.append(prompt)

for message in chat_history:
    print(message)

current_timestamp = datetime.utcnow().isoformat().split('.')[0] + "Z"

response_message = completion["choices"][0]["message"]["content"]

chat_history.append({'role': 'assistant', 'content': response_message, 'name': 'geist'})

print(response_message)

save_chat_history(chat_history_path, chat_history)