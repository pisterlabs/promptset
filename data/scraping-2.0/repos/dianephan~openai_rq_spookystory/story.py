from dotenv import load_dotenv
import os
from random import choice
import openai
from flask import Flask, request

load_dotenv()
openai.api_key = os.environ.get('OPENAI_KEY')
completion = openai.Completion()

session_prompt = "The following is a spooky story written for kids, just in time for Halloween. We went to the playground after school, but the swing on the swingset kept swinging on its own."

def write_story(session_story=None):
    file1 = open("spookystory.txt","a") 
    if session_story == None: 
        prompt_text = session_prompt
        file1.write(session_prompt)
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
    print(story)
    append_to_story(story, session_story)
    file1.write(story)
    return str(story)

def append_to_story(story, session_story=None):
    # print("Appended to story") 
    if session_story is None:
        session_story = session_prompt
    return f'{session_story}{story}'