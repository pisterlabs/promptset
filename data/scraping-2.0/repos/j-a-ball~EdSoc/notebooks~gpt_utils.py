__author__ = "Jon Ball"
__version__ = "October 2023"

import openai
import tiktoken
#import time
import os
from dotenv import load_dotenv
_ = load_dotenv("openai.env")
openai_key = os.environ.get("OPENAI_KEY")
openai.api_key = openai_key

def get_completion(messages, temp=0, model="gpt-4"):
    #tok = time.time()
    #print(f"{model} generating completion...")
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        max_tokens=75,
        seed=42,
        temperature=temp, # this is the degree of randomness of the model's output
    )
    #tik = time.time()
    #print(f"   ...completion generated in {round(tik-tok, 2)} seconds.")
    return response.choices[0].message["content"]

encoding = tiktoken.encoding_for_model("gpt-4")
def len_messages(messages, encoding=encoding):
    toks = 0
    for m in messages:
        toks += len(encoding.encode(m["content"]))
    print(f"{toks} tokens in message cache of len {len(messages)}.\n")

def start_chat(system_role):
    messages = []
    messages.append(
        {"role": "system", "content": system_role})
    return messages

def user_turn(messages, user_message):
    messages.append(
        {"role": "user", "content": user_message})
    #len_messages(messages)
    return messages

def system_turn(messages, temp=0, model="gpt-4"):
    system_message = get_completion(messages, temp=temp, model=model)
    messages.append(
        {"role": "system", "content": system_message})
    #print(f"System Response:\n\n {system_message}")
    #len_messages(messages)
    return messages