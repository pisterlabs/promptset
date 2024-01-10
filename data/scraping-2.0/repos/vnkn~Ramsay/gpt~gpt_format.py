import json
import os

import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')

def call_gpt_func(functions, messages, model="gpt-3.5-turbo-0613", tries=3):
    response = openai.ChatCompletion.create(
        # model="gpt-4-0613",
        # model="gpt-3.5-turbo-0613",
        model=model,
        functions=functions,
        messages=messages,
    )
    args = response.choices[0].message.function_call.arguments
    args = json.loads(args)
    if not args['ingredients'] and tries > 0:
        print('GPT fail!!')
        return call_gpt_func(functions, messages, model=model, tries=tries-1)
    return args

PARAMS = {
    "type": "object",
        "properties": {
            "vegan": {
                "type": "boolean",
                "description": "Is this recipe vegan?"
            },
            "appliances": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "What are the appliances required?"
            },
            "appliance_costs": {
                "type": "array",
                "items": {
                    "type": "integer"
                },
                "description": "How much does each appliance cost?"
            },
            "ingredients": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "What are the ingredients required?"
            },
            "ingredient_costs": {
                "type": "array",
                "items": {
                    "type": "integer"
                },
                "description": "How much does each ingredient cost?"
            },
            "steps": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "What are the steps, in chronological ordering?"
            },
            "serves": {
                "type": "integer",
                "description": "How many people does it serve?"
            },
            "time": {
                "type": "integer",
                "description": "How long does it take to make?"
            },
            "title": {
                "type": "string",
                "description": "What is the title of the dish?"
            },
        },
    "required": ["vegan", "appliances", "appliance_costs", "ingredients", "ingredient_costs", "steps", "serves", "time", "title"]
}

def format_web_clean(text: str):
    functions = [
        {
            "name": "extract_data_from_recipe",
            "description": "Given a recipe, extract the given fields.",
            "parameters": PARAMS,
        }
    ]
    messages = [
        {"role": "system", "content": "You are a helpful assistant that parses a cooking recipe, and extracts relevant information."},
        {"role": "user", "content": 'Here is the recipe: ' + text}
    ]
    return call_gpt_func(functions, messages)


def format_web_dirty(text: str):
    functions = [
        {
            "name": "extract_data_from_recipe",
            "description": "Extract the given fields from the webscraped text.",
            "parameters": PARAMS,
        }
    ]
    messages = [
        {"role": "system", "content": "You are a helpful assistant that parses a webscraped page's text, and extracts relevant recipe information. This involves ignoring the irrelevant information and focusing only on the relevant recipe details."},
        {"role": "user", "content": 'Here is the webscraped text: ' + text}
    ]
    return call_gpt_func(functions, messages)


def format_yt(transcript: str) -> dict:
    functions = [
        {
            "name": "extract_data_from_transcript",
            "description": "Given the transcript of a YouTube video, extract the given fields.",
            "parameters": PARAMS
        }
    ]

    messages = [
        {"role": "system", "content": "You are a helpful assistant that parses a cooking YouTube video transcript, and extracts relevant information."},
        {"role": "user", "content": 'Here is the transcript: ' + transcript}
    ]

    return call_gpt_func(functions, messages)