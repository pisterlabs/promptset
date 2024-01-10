from dotenv import load_dotenv
import os
from random import choice
import openai
from flask import Flask, request

load_dotenv()
openai.api_key = os.environ.get('OPENAI_KEY')
completion = openai.Completion()

start_sequence = "\nFun description:"
restart_sequence = "\n\nThe tags for this picture are:"
session_prompt="""
Below are some witty fun descriptions for Instagram pictures based on the tags describing the pictures.

The tags for this picture are: { }
Fun description: There is no picture.

The tags for this picture are:  {'food': 1, 'sweet': 1, 'chocolate': 1, 'sugar': 1, 'cake': 1, 'milk': 1, 'delicious': 1, 'cup': 1, 'candy': 1, 'no person': 1, 'breakfast': 1, 'baking': 1, 'party': 1, 'cream': 1, 'vacation': 1, 'Christmas': 1, 'coffee': 1, 'table': 1, 'color': 1, 'cookie': 1}
Fun description: Loving this delicious Christmas dessert platter this year! Happy Holidays everyone! 

The tags for this picture are:  {'food': 1, 'sweet': 1, 'chocolate': 1, 'sugar': 1, 'cake': 1, 'milk': 1, 'delicious': 1, 'cup': 1, 'candy': 1, 'no person': 1, 'breakfast': 1, 'baking': 1, 'party': 1, 'cream': 1, 'vacation': 1}
Fun description: Took myself on vacation to enjoy some fancy chocolate. A girl's best friend!

The tags for this picture are:  {'food': 1, 'sweet': 1, 'chocolate': 1, 'sugar': 1, 'cake': 1, 'milk': 1, 'delicious': 1, 'breakfast': 1, 'baking': 1, 'party': 1, 'cream': 1, 'vacation': 1}
Fun description: A perfectly small cake that I baked for my friends birthday!

The tags for this picture are:  {'hot': 1, 'cheetos': 1, 'snack': 1, 'yummy': 1, 'junk': 1, 'food': 1, 'delicious': 1, 'vacation': 1}
Fun description: I was so delighted when I found these cheetos that tasted exactly like pepperoni pizza!
"""

def generate_caption(picture_tags):
    prompt_text = f'{session_prompt}{restart_sequence}: {picture_tags}{start_sequence}:'
    response = openai.Completion.create(
      engine="davinci",
      prompt=prompt_text,
      temperature=0.7,
      max_tokens=64,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0.3,
      stop=["\n"],
    )
    caption = response['choices'][0]['text']
    return str(caption)
