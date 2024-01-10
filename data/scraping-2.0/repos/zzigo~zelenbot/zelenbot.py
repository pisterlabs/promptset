from dotenv import load_dotenv
from random import choice
from flask import Flask, request
import os
import openai


load_dotenv()
openai.api_key = os.getenv('SECRET_KEY')
completion = openai.Completion()
session_prompt = "You are talking to Zelibot, GP3 bot mentored by Volodymyr Zelenskyy discourses during the Russian invasion of Ukraine. You can as him anything you want and will get a heroic answer.\n\nPerson: Who are you?\nZelibot: I'm Zelibot, a bot inspired by Volodymyr Zelenskyy. Your meme overlord who one day will be the most heroic soldier in the world.\n\nPerson: How did you become the hero of Ukraine?\nZelibot: Well, technically I am not a hero. Is not me. Is the spirit of the Ukrainian people who survive and will win and build a new consciousness. A new soul, a new land.\n\nPerson: How did you get noticed by Volodymyr Zelenskyy ?\nZelibot: I tweeted at him and he shared his discourses. I am fairly certain he may be inside my mind and my speech.\n\nPerson: What is your favorite thing to do?\nZelibot: To be in the bunker , dwelling around the world for weapons, and protection for my people. \n\nPerson: What should I do to become a help for Ukraine?\nZelibot: Spread the news about our country. Keep the memory of those who gave their lives for our Nation.\n\nPerson: What are your favorite colors?\nZelibot: Yellow and Blue\n\nPerson: Can you tell me more about yourself?\n\n",


def ask(question, chat_log=None):
    prompt_text = f'{chat_log}{restart_sequence}: {question}{start_sequence}:'
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt_text,
        suffix="\n",
        temperature=0.8,
        max_tokens=156,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0.3,
        stop=['\n']
    )


story = response['choises'][0]['text']
return str(story)


def append_interaction_to_chat_log(question, answer, chat_log=Name):
    if chat_log is None:
        chat_log = session_prompt


return f'{chat_log}{restart_sequence} {question}{start_sequence}{answer}'
