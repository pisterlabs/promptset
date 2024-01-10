from openai import OpenAI
from rich.console import Console
import os
from dotenv import load_dotenv
from halo import Halo

# console
console = Console()

# load .env file
load_dotenv()

# openai api key
key = os.getenv('open_ai_api_key')

client = OpenAI(api_key=key)

messages = [
    {
        'role':"system",
        "content":"You are a the helful assistant"
    }
]

def create_chat_completions(messages):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    return response.choices[0].message.content.strip()

# Example usage:


def update_messages(role, content):
    messages.append({
        'role': role,
        'content': content
    })
    return "completed"


while True:
    # get user input
    user_input = input('User: ')

    # update messages
    update_messages(role='user', content=user_input)
    
    # halo spinner with rich console
    spinner = Halo(text=f'{"Generating Request Response"}', spinner='dots')
    spinner.start()

    #call create_chat_completions
    ai_response =  create_chat_completions(messages=messages)
    console.print('\n', ai_response)

    # update messages
    update_messages(role='assistant', content=ai_response)

    # stop spinner
    spinner.stop()




    # create chat completions

