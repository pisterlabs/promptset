import os

import openai
import json

from dotenv import load_dotenv, find_dotenv
from icecream import ic

load_dotenv(find_dotenv())

openai.api_key = os.getenv("OPENAI_API_KEY")

# this is good example of function calling in openAI - created by Emil K - source aidevs2 forum
# small modifications by Dominik Jeziorski

# definition of multiply function
def mult(second, first):
    x = second * first
    ic(f'Multiply results: {x}')
    return x


# definition of adding function
def adding(second, first):
    x = second + first
    ic(f'Adding results: {x}')
    return x


# function_calling structure
functions = [
    {
        "name": "adding",
        "description": "Adding two numbers",
        "parameters": {
            "type": "object",
            "properties": {
                "first": {
                    "type": "integer",
                    "description": "First of the integer",
                },
                "second": {
                    "type": "integer",
                    "description": "First of the integer",
                },
            },
            "required": ["first", "second"],
        },
    },
    {
        "name": "mult",
        "description": "Multiply two numbers",
        "parameters": {
            "type": "object",
            "properties": {
                "first": {
                    "type": "integer",
                    "description": "First of the integer",
                },
                "second": {
                    "type": "integer",
                    "description": "First of the integer",
                },
            },
            "required": ["first", "second"],
        },
    }
]

# user prompt
messages = [
    {"role": "user", "content": "Please add 10 to 52"},
]

# response of function_calling prompt
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0613",
    messages=messages,
    functions=functions,
    function_call="auto",  # auto is default, but we'll be explicit
)

# ic the message
response_message = response["choices"][0]["message"]
ic(response_message)

# if function_call response
if response_message.get("function_call"):
    # available functions to choose
    available_functions = {
        "adding": adding,
        "mult": mult,
    }

    # get the name of the function to call
    fname = response_message["function_call"]["name"]
    ic(fname)
    function_to_call = available_functions[fname]

    # get arguments of the function
    function_args = json.loads(response_message["function_call"]["arguments"])
    ic(function_args)

    # call the chosen function
    function_response = function_to_call(function_args["first"], function_args["second"])
