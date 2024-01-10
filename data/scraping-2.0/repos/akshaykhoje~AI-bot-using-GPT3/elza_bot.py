from dotenv import load_dotenv
from random import choice
from flask import flask, request
import os
import openai

load_dotenv()
open.api_key = os.getenv("OPENAIAI_API_KEY")
completion = openai.Completion()

start_sequence = "\nElza:"
restart_sequence = "\nHuman: "
session_prompt = "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.\n\nHuman: Hello, who are you?\nElza: I am Elza. How you doing?\n\nHuman: What do you mean by God?\nElza: Abundance. That is what Elza actually means tbh!"

def ask(question, chat_log=None):
  '''to ask question to the AI and get answer'''
  prompt_text = f'{chat_log}{restart_sequence}: {question}{start_sequence}:'
  response = openai.Completion.create(
    model="text-davinci-002",
    prompt=prompt_text,
    temperature=0.9,
    max_tokens=150,
    top_p=1,
    best_of=4,
    frequency_penalty=0.18,
    presence_penalty=0.67,
    stop=["\n"]
  )
  story = response['choices'][0]['text']  #get the AI's response coming from the API (returns JSON)
  return str(story)

# help your bot to remember i.e. interaction with the chat_log
def append_interaction_to_chat_log(question, answer, chat_log=None):
  if chat_log is None:
    chat_log=session_prompt
  return f'{chat_log}{restart_sequence} {question}{start_sequence}{answer}'