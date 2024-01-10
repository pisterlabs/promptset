# Description: This file contains the functions that interact with the GPT-3.5 model
# For the lab, you don't need to change anything in this file
from constants import OPEN_AI_API_KEY, BASELINE_MESSAGES
import openai

openai.api_key = OPEN_AI_API_KEY
if not openai.api_key:
    raise Exception("Please enter your openai api key in constants.py")

def clean_print(text):
    # remove starting and ending blank characters and quotes
    text.content = text.content.strip().strip('"')
    print("Content:\n", text.content)

def process_result(result, conversation = [], save = True):
    msg = result.choices[0].message
    if save:
        conversation.append(msg)
    return msg

def make_message(message, role = "user"):
    return {"role": role, "content": message}

#TODO: Step 4. Implement get_response, which takes in a message and coversation, and uses GPT 3.5 turbo to return an input to process_result()
def get_response(message, conversation = []):
    # return process_result( , )
    return

def save_conversation(conversation, filename = "output.txt"):
    with open(filename, "w") as f:
        for msg in conversation:
            f.write(msg["role"] + ": " + msg["content"] + "\n")
    

