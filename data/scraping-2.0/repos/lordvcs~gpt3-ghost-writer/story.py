from dotenv import load_dotenv
import os
from random import choice
import openai
from flask import Flask, request

load_dotenv()
openai.api_key = os.environ.get('OPENAI_KEY')
completion = openai.Completion()

session_prompt = "The following is a spooky story written for kids, just in time for Halloween. Everyone always talks about the old house at the end of the street, but I couldn't believe what happened when I went inside."

def write_story(session_story=None):
    if session_story == None: 
        prompt_text = session_prompt
    else:
        prompt_text = session_story
    response = openai.Completion.create(
      engine="davinci",
      prompt=prompt_text,
      temperature=0.7,
      max_tokens=96,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0.3,
    )
    story = response['choices'][0]['text']
    return str(story)


def append_to_story(story, session_story=None):
    if session_story is None:
        session_story = session_prompt
    return f'{session_story}{story}'