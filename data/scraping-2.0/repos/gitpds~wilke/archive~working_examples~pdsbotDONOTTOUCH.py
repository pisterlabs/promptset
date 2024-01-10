from dotenv import load_dotenv
from random import choice
from flask import Flask, request
import os
import openai
import datetime

datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")


load_dotenv()
open_ai_key = os.getenv("OPENAI_API_KEY")
completion = openai.Completion()

start_sequence = "\nAI:"
restart_sequence = "\nHuman: "
session_prompt = "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly."

def ask(question, chat_log=None):
    prompt_text = f'{chat_log}{restart_sequence}{question}{start_sequence}'
    
    response = openai.Completion.create(
    model="text-davinci-003",
    # model="davinci:ft-pds:plot-generator-2022-12-23-20-13-29",
    prompt= prompt_text,
    temperature=0.9,
    max_tokens=150,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0.6,
    stop=[" Human:", " AI:"]
    )
    story = response['choices'][0]['text']
    return str(story)



def append_interaction_to_chat_log(question, answer, chat_log=None):
    if chat_log is None:
        # Get the current date and time to use as the file name
        
        # current_time = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        # Get the current time
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        # Create the logs folder if it doesn't already exist
        os.makedirs('logs', exist_ok=True)
        # Create the chat log file with the current time as the name
        chat_log = open(f'logs/chat_log_{current_time}.txt', 'w')
    chat_log.write(f'{question}{start_sequence}{answer}')
    return chat_log