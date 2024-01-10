import base64
import datetime
import json
import os

import openai
import requests
import vosk
from dotenv import load_dotenv
from flask import Flask, request
from flask_sock import ConnectionClosed, Sock
from twilio.rest import Client
from twilio.twiml.voice_response import Start, VoiceResponse

load_dotenv()
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler                                  
llm = Ollama(model="mistral:7b-instruct")

url = "http://127.0.0.1:5000/make_call"


user_prompt = "Question: " + input("Enter your prompt: ")
messages: list[dict[str, str]] = [
    {"role": "user", "content":
     "Output one word containing LOCATION for the location of the spot where the user wants to order pizza from. The possible LOCATION are: 'PizzaPizza', 'PizzaNova', 'Dominos'. Make sure you keep capitalization and type exactly as before! Do not output anything else. Do not output quotes, or newlines, or location: XXX. Just output one word."},
    {"role": "user", "content": user_prompt},
    {"content": "What LOCATION does the user want?"}
]

phone_numbers = {
    'pizzanova': '+18443103300',
    'dominos': '+15197452222',
    'pizzapizza': '+15197471111',
}



# response = openai.ChatCompletion.create(
#     model="gpt-3.5-turbo",
#     messages=messages
# )

lc = '\n'.join([x['content'] for x in messages])
print("Prompt", lc)
response = llm(lc)
location = response.lower().strip('"')

# location = response["choices"][0]["message"]["content"]
phone = phone_numbers[location]
print(response)
print(url + f'?location={location}&phone={phone}')
response = requests.get(url + f'?location={location}&phone={phone}')
print(response.content)
