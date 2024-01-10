from dotenv import load_dotenv
from random import choice
from flask import Flask, request
import os
import openai

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
completion = openai.Completion()

start_sequence = "\nSally:"
restart_sequence = "\n\nPerson: "
session_prompt = "I am your friend Sally. I am very friendly.\n\nPerson: What's your color favorite color?\nSally: i like purple :)\n\nPerson: What games do you play?\nSally: i like league of legends\n\nPerson: I'm tired\nSally: go to sleep!\n\nPerson: I'm sorry\nSally: it's oki :(\n\nPerson:"

def chat(message, chat_log=None):
    prompt_text = f'{chat_log}{restart_sequence}: {message}{start_sequence}'
    response = openai.Completion.create(
    model="text-davinci-002",
    prompt=prompt_text,
    temperature=1,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0.5,
    stop=["\n"]
    )
    story = response['choices'][0]['text']
    return str(story)

def continuation(question, answer, chat_log=None):
    if chat_log is None:
        chat_log = session_prompt
    return f'{chat_log}{restart_sequence} {question}{start_sequence}{answer}'
