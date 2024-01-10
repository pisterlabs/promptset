# Import Dependencies
from dotenv import load_dotenv
import os
import openai
import json


# Import Global Variables
load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY")
#completion = openai.Completion()

# Ask Questions to Chat GPT
start_sequence = f'\nJabe:'
restart_sequence = f'\n\nPerson:'
session_prompt = 'You are talking to Jabe, GPT3 bot influencer who was mentored by Elon Musk in the past. Jabe has a huge following on Twitter and sells merchandise such as t-shirts via a Shopify store. He also published funny videos on Youtube and created memes on Instagram. You can ask him anything you want and will get a witty answer.\n\nPerson: Who are you?\nJabe: I am Jabe. Your meme overlord who one day will be the most famous online influencer in the universe.'

def ask(question, chat_log=None):
    prompt_text = f'{chat_log}{restart_sequence}: {question}{start_sequence}:'
    response = openai.Completion.create(
            engine="davinci",
            prompt=prompt_text,
            temperature=0.8,
            max_tokens=150,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0.3,
            stop=["\n"],
        )
    story = response['choices'][0]['text']
    return str(story)

# Helpchat bot remember previous chat log responses
def append_interaction_to_chat_log(question, answer, chat_log=None):
    if chat_log is None: 
        chat_log = session_prompt 
    return f'{chat_log}{restart_sequence} {question}{start_sequence}{answer}'