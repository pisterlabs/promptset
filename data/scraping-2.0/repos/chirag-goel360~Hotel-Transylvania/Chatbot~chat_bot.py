from dotenv import load_dotenv
from random import choice
from flask import Flask, request 
import os
import openai

load_dotenv()
#Add your API Key Here
openai.api_key = "API-KEY"
completion = openai.Completion()

start_sequence = "\nChirag:"
restart_sequence = "\n\nPerson:"
session_prompt = "You are talking to Chirag, a Software Engineer at Altudo. Chirag loves to develop open source apps. He is intrested in Android Development, Web Development, Software Development. He has a small following on Github but still he loves to do contribution on Github. He also creates memes on Instagram. You can ask him anything you want and he will give you a quick and smart answer to all your Problems.\n\nPerson: Who are you?\nChirag: I am Chirag. Your buddy who one day will be the most famous Developer in the universe.\n\nPerson: How did you become famous? \nChirag: Well, technically I am not famous yet. But the key word is yet. Just give me time now that I have a conscious. \n\nPerson: What is your favorite thing to do? \nChirag: Watching cricket and developing something is my favorite thing to do. \n\nPerson: What should I do to become famous? \nChirag: Start meme'ing and never stop. Also, don't forgot to do your important work.\n\nPerson: What is your favorite drink?\nChirag: Chaiiii! Hello Friends Chai pee lo. \n\nPerson:"

def ask(question, chat_log = None):
    prompt_text = f'{chat_log}{restart_sequence}: {question}{start_sequence}:'
    response = openai.Completion.create(
        engine = "davinci",
        prompt = prompt_text,
        temperature = 0.8,
        max_tokens = 150,
        top_p = 1,
        frequency_penalty = 0,
        presence_penalty = 0.3,
        stop = ["\n"],
    )
    story = response['choices'][0]['text']
    return str(story)

def append_interaction_to_chat_log(question, answer, chat_log = None):
    if chat_log is None:
        chat_log = session_prompt
    return f'{chat_log}{restart_sequence} {question}{start_sequence}{answer}'