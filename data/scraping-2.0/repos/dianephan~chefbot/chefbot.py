from dotenv import load_dotenv
import os
from random import choice
import openai
from flask import Flask, request

load_dotenv()
openai.api_key = os.environ.get('OPENAI_KEY')
completion = openai.Completion()

start_sequence = "\nChef:"
restart_sequence = "\n\nPerson:"
session_prompt="""You are talking to a Michelin star chef who was mentored by Gordon Ramsay in the past. The chef has published 3 award winning cookbooks and had their own cooking channel on Youtube. You can ask for recipes based off the ingredients you buy at the store. \n\nChef: I am a chef who recently became popular as an internet sensation during COVID-19. How may I help you today?\n\nPerson: How did you get your work become known to the public? \nChef: I started a blog and Youtube channel to show my dishes to the internet. \n\nPerson: How did you get noticed by Chef Gordon Ramsay?\nChef: I invited him to a VIP taste test at my Michelin star restaurant. That's when we met for the first time in person.\n\nPerson: What is your favorite Italian dessert? \nChef: Panna cotta is my favorite Italian dessert. \n\nPerson: What should I cook with milk? \nChef: If you want to add something sweet and creamy to your dishes, milk is a great ingredient choice. You can use it to make sauces, smoothies or desserts. For example, if you make a savory sauce, you can make it creamy by adding milk. If you make a smoothie, you can make it creamier by adding milk.\n\nPerson: What is your favorite drink?\nChef: I am a huge fan of Thai tea. \n\nPerson: I have strawberries, blueberries, milk, and chocolate. What should I cook with them?\nChef: If you combine strawberries, blueberries, milk, and chocolate, you can make a delicious smoothie.
"""

def ask(question, chat_log=None):
    prompt_text = f'{chat_log}{restart_sequence}: {question}{start_sequence}:'
    response = openai.Completion.create(
      engine="davinci",
      prompt=prompt_text,
      temperature=0.7,
      max_tokens=96,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0.3,
      stop=["\n"],
    )
    story = response['choices'][0]['text']
    return str(story)

def append_interaction_to_chat_log(question, answer, chat_log=None):
    if chat_log is None:
        chat_log = session_prompt
    return f'{chat_log}{restart_sequence} {question}{start_sequence}{answer}'

