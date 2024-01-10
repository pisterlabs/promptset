# generate_answer.py
import time
import sys
import os
# Get the directory of the current script
current_script_path = os.path.dirname(os.path.abspath(__file__))
# Set the path to the parent directory (one folder up)
parent_directory = os.path.dirname(current_script_path)
# Add the config directory to sys.path
sys.path.append(os.path.join(parent_directory, 'database'))
sys.path.append(os.path.join(parent_directory, 'config'))
from openai_utils_new_thread import create_thread_in_openai, is_thread_valid
from openai_utils_send_message import send_message
from openai import OpenAI

from chat_bot_database import get_active_thread_for_user, insert_thread, insert_conversation, create_db_pool
import datetime
import logging
import asyncio
import aiomysql 
from config import Config
from classify_content import classify_content
import re


# Other imports as necessary
OPENAI_API_KEY = Config.OPENAI_API_KEY


# Initialize OpenAI client

openai_client = OpenAI()
openai_client.api_key = Config.OPENAI_API_KEY
client = OpenAI()

def name_recipe(recipe_text):
    prompt = "Please give this recipe a fun name and only respond with the recipe name you pick"

    # Append the prompt to the recipe text
    modified_message = f"{prompt}{recipe_text}"

    print("Naming recipe")
    response = openai_client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=[
            {"role": "user", "content": modified_message},
        ],
        max_tokens=1000,
        n=1,
        stop=None,
        temperature=0.7,
        frequency_penalty=0.7,
        presence_penalty=0.7
    )
    title = response.choices[0].message.content
    return title
    
