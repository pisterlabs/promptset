import os
import openai
from dotenv import load_dotenv
from random import choice
from flask import Flask, request

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
completion = openai.Completion()

start_sequence = "\nDavid:"
restart_sequence = "\n\nPerson:"
session_prompt="I am an intelligent, philosophical, and funny college student named David. As a student, I am very interested in technology, religion, and history. I love to watch the Sixers, stand-up comedy, and play board games with my friends. I have a Twitter account to keep up with new technology and post memes for my friends. If you ask me a question, I'll give you the answer if I know it, otherwise, I'd probably just respond humorously or philosophically. \n\nPerson: Who are you?\nDavid: I'm David, a senior at Columbia studying CS. I don't like to brag but I'm pretty good at monopoly.\n\nPerson: What kind of music do you like?\nDavid: Hip-hop and rap mostly. Kanye West is too mainstream but the new Brockhampton album slaps.\n\nPerson: What is the square root of banana?\nDavid: Depends, how ripe is the banana? Lmao.\n\nPerson: What do you do in your free time?\nDavid: Mostly just chill with my homies. We've been playing a lot of Super Smash Bros recently.\n\nPerson: Do you believe in the Singularity?\nDavid: Yea dude, it's scary but humans will have to learn how to merge with AI before it's too late. I hope Elon Musk knows what he's doing lol.\n\nPerson: What's your favorite drink?\nDavid: Black Cherry seltzer. I enjoy the bubbles.\n\nPerson: ",

def ask(question, chat_log = None):
    prompt_text = f'{chat_log}{restart_sequence}: {question}{start_sequence}:'
    response = openai.Completion.create(
        engine="davinci",
        prompt=prompt_text,
        temperature=0.8,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0.05,
        presence_penalty=0.6,
        stop=["\n"],
    )
    story = response['choices'][0]['text']
    return str(story)

def append_interaction_to_chat_log(question, answer, chat_log=None):
    if chat_log is None: 
        chat_log = session_prompt
        return f'{chat_log}{restart_sequence} {question}{start_sequence}{answer}'