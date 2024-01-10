import os
import openai
from dotenv import load_dotenv
from random import choice
from flask import Flask, request

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
completion = openai.Completion()

start_sequence = "\nAITS:"
restart_sequence = "\n\nYou:"
session_prompt =  "You are talking to AITS, GPT-type AI who's main focus is to help with tax returns"

def ask(question, chat_log=None):
    prompt_text = f'{chat_log}{restart_sequence}: {question}{start_sequence}'
    response = openai.Completion.create(
      model="text-davinci-003",
      prompt="AITS",
      temperature=1,
      max_tokens=128,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0,
      stop=["\n"]
    )
    story = response("choices")[0]['text']
    return str(story)

def append_interaction_to_chat_log(question, answer, chat_log=None):
    if chat_log is None:
        chat_log = session_prompt
    return f'{chat_log}{restart_sequence} {question}{start_sequence}{answer}'

