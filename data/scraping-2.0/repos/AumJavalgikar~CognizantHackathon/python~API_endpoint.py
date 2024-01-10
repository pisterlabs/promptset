import random
import uuid

from fastapi import FastAPI
import os
from dotenv import load_dotenv
import openai
from openai import AsyncOpenAI
import yaml
import json
import re

load_dotenv('secrets/secrets.env')

app = FastAPI()

client = AsyncOpenAI(
    api_key=os.getenv('OPENAI_KEY'),
)


@app.get("/")
async def test():
    response = await client.chat.completions.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {"role": "system", "content": 'This is a test, respond with "GPT works"'},
            {"role": "user", "content": ''},
            {"role": "assistant", "content": ''}
        ]
    )
    conversation_response = response.choices[0].message.content
    return conversation_response


@app.get("/getConversationResponse/")
async def getConversationResponse(conversation_info: dict):
    '''
    conversation_info format :
    {
    'current_speaker' : 'character1_name',
    'other_speaker': 'character2_name',
    'current_time' : 'time',
    'new_response' : 'new_response' # Needed for user sent messages
    conversation_initiated : True or False
    conversation_initiate_reason : 'reason to start conversation'
    }
    '''

    current_speaker = conversation_info.get('current_speaker')
    other_speaker = conversation_info.get('other_speaker')
    current_time = conversation_info.get('current_time')
    new_response = conversation_info.get('new_response')
    conversation_initiated = conversation_info.get('conversation_initiated')
    conversation_history = get_conversation_history(current_speaker, other_speaker)
    if conversation_initiated:
        conversation_initate_reason = conversation_info.get('conversation_initate_reason')
        last_response = f'''
        You decided to initiate conversation with {other_speaker} for the following reason : {conversation_initate_reason}\n
        Say something using the given syntax.
        '''
    else:
        last_response = f'''
        {other_speaker} said '{new_response}'
        Respond to this using the given syntax.
        '''

    with open('prompts/conversation/base_prompt.yaml') as f:
        base_prompt = yaml.safe_load(f)
        system_prompt = base_prompt.get('system')
        user_prompt = base_prompt.get('user')
        system_prompt = system_prompt.format(current_speaker_details=get_character_details(current_speaker),
                                             other_speaker=other_speaker,
                                             other_speaker_details=get_character_details(other_speaker))
        user_prompt = user_prompt.format(last_response=last_response,
                                         current_speaker=current_speaker,
                                         conversation_history=conversation_history
                                         )

    print(system_prompt)
    print(user_prompt)

    response = await client.chat.completions.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": ""}
        ]
    )
    print(response.choices[0].message.content)
    regex = re.compile('({.*})')
    json_response = regex.findall(response.choices[0].message.content)[0]
    conversation_response = json.loads(json_response, strict=True)

    update_conversation_history(current_speaker, other_speaker, current_time, conversation_response, new_response)

    return conversation_response


@app.get("/getAction/")
async def getAction(action_info: dict):
    '''
    conversation_info format :
    {
    'character_name' : 'character1_name',
    'current_time': 'time',
    'other_character_status' : 'statuses'
    }
    '''

    character_name = action_info.get('character_name')
    current_time = action_info.get('current_time')
    other_character_status = action_info.get('other_character_status')
    journal = get_journal(character_name)
    last_location = action_info.get('last_location')
    last_action = action_info.get('last_action')

    with open('prompts/actions/base_prompt.yaml') as f:
        base_prompt = yaml.safe_load(f)
        system_prompt = base_prompt.get('system')
        user_prompt = base_prompt.get('user')
        system_prompt = system_prompt.format(current_time=current_time,
                                             other_character_status=other_character_status,
                                             current_speaker_details=get_character_details(character_name),
                                             journal=journal)
        user_prompt = user_prompt.format(character_name=character_name,
                                             random_seed=random.randint(0, 10000),
                                         last_location=last_location,
                                         last_action=last_action)

    print(system_prompt)
    print(user_prompt)

    response = await client.chat.completions.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": ""}
        ]
    )
    print(response.choices[0].message.content)
    regex = re.compile('({.*})')
    json_response = regex.findall(response.choices[0].message.content)[0]
    conversation_response = json.loads(json_response, strict=True)

    update_journal(character_name, conversation_response, time_stamp=current_time)

    return conversation_response


def get_journal(character_name):
    try:
        with open(f'journals/{character_name}', 'r') as f:
            pass
    except FileNotFoundError:
        with open(f'journals/{character_name}', 'w') as f:
            f.write(f'Journal for {character_name}\n')

    with open(f'journals/{character_name}', 'r') as f:
        journal = f.read()

    return journal


def update_journal(character_name, action, time_stamp):
    new_entry = action['journal_entry']

    with open(f'journals/{character_name}', 'a') as f:
        f.write(f'\n{time_stamp} : {new_entry}')


def get_conversation_history(char1, char2):
    # Check if conversation exists, if not, create a new conversation
    try:
        with open(f'conversations/{char1}/{char2}', 'r') as f:
            pass
    except FileNotFoundError:
        with open(f'conversations/{char1}/{char2}', 'w') as f:
            f.write(f'Conversation history for {char1} and {char2}\n')

        with open(f'conversations/{char2}/{char1}', 'w') as f:
            f.write(f'Conversation history for {char2} and {char1}\n')

    with open(f'conversations/{char1}/{char2}', 'r') as f:
        history = f.read()

    return history


def update_conversation_history(char1, char2, timestamp, new_response, old_response):
    new_response = new_response['conversation_response']
    with open(f'conversations/{char1}/{char2}', 'a') as f:
        f.write(f'\n{timestamp} {char2} : {old_response}')
        f.write(f'\n{timestamp} {char1} : {new_response}')

    with open(f'conversations/{char2}/{char1}', 'a') as f:
        f.write(f'\n{timestamp} {char2} : {old_response}')
        f.write(f'\n{timestamp} {char1} : {new_response}')


def get_character_details(character):
    with open(f'prompts/character_details/{character}.yaml', 'r') as f:
        details = yaml.safe_load(f)
    return details['character_details']
