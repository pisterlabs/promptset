import asyncio
import inspect

import openai

import common.utils as utils
from db.datastore import db
from llm.prompt import extract_keywords_prompt, get_random_memories_prompt


async def get_chat_response(prompt):
    messages = [{"role": "user", "content" : prompt}]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages=messages
    )
    result = response['choices'][0]['message']['content']
    return result

async def get_chat_response_dictionary(prompt):
    try:
        result = await get_chat_response(prompt)
        lines = result.splitlines()
        dictionary = {}
        for line in lines:
            if '::' not in line:
                continue # Skip lines that don't have a colon, which happens when the llm decides to add something unnecessary
            key, value = line.split('::')
            dictionary[key.strip().lower()] = value.strip()
        return dictionary
    except Exception as e:
        print("Error in parsing get_chat_response_dictionary:\n" + result)
        utils.print_error(inspect.currentframe().f_code.co_name, e)   

async def extract_keywords(input):
    prompt = extract_keywords_prompt(input)
    return await get_chat_response(prompt)

async def add_random_memories(entity_id, count):
    prompt = get_random_memories_prompt(count)
    response = await get_chat_response(prompt)
    memories = response.splitlines()
    memories = [strip_non_letters(memory) for memory in memories]
    db.save_memories(entity_id, 1, memories)