#!/usr/bin/python3

import os
import openai
import json

openai.api_key = os.getenv("OPENAI_API_KEY")

def start_prompt():
    """Reads the start prompt from the start prompt json file"""
    with open("start_prompt.json", "r") as f:
        prompt = json.load(f)
    return prompt


def create_new_prompt(prev_prompt, prompt: str):
    """Creates a valid gpt-3.5 prompt from users message"""
    new_prompt = [
        prev_prompt,
        {"role" : "user", "content" : prompt}
        ]
    prev_prompt = new_prompt[1]
    return new_prompt


def start_new_topic(subject="default"):
    """Gives chatbot detail of what to do on request of new topic"""
    

def call_bot(start_prompt):
    """creates calls bot and initiate conversation conversation"""
    completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages = start_prompt
            )
    return dict(completion.choices[0].message)

if __name__ == "__main__":
    #print(call_bot(start_prompt()).get("content"))
    prev_prompt = start_prompt()[0]
    print(call_bot(create_new_prompt(prev_prompt, "Tell me a funfact")).get("content"))
    print(prev_prompt)
    print(call_bot(create_new_prompt(prev_prompt, "Wow, Nice Tell me another one")).get("content"))
    exit(0)
