from modularity import flatten_whitespace, indent
import json
from pymongo import MongoClient
from modularity import OpenAI
import traceback
import datetime as dt
from bson import ObjectId as oid

db = MongoClient()['DMLLM']
openai = OpenAI()

#DEFAULT_MODEL = "gpt-4-1106-preview"
DEFAULT_MODEL = "gpt-3.5-turbo"

def get_response(messages, model=DEFAULT_MODEL, **kwargs):
    messages = [{'role': m['role'], 'content': m['content']} for m in messages]
    kwargs = {
        'max_tokens': None,
        'temperature': 0.9,
        'top_p': 1,
        'frequency_penalty': 0.0,
        'presence_penalty': 0.6,
        **kwargs,
    }

    response = openai.chat.completions.create(
        model=model,
        messages = messages,
        **kwargs,
    )

    return response.choices[0].message.content

def json_retry_loop(messages, model=DEFAULT_MODEL, loop_i=0):
    while True:
        response = get_response(messages, model=model)
        try:
            return json.loads(response)
        except json.decoder.JSONDecodeError:
            messages.append({'role': 'system', 'content': "Invalid JSON. Please try again."})

            loop_i += 1
            if loop_i > 3:
                raise
            
            return json_retry_loop(messages, model=model, loop_i=loop_i)

import logging
import sys

def setup_logging():
    # Create a handler that writes log messages to sys.stdout
    stdout_handler = logging.StreamHandler(sys.stdout)

    # Set the format for the handler
    stdout_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                                  datefmt='%Y-%m-%d %H:%M:%S'))

    # Get the root logger and set its handler and level
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)
    root_logger.addHandler(stdout_handler)
    
def enable_logging_for_submodules(submodules, level=logging.DEBUG):
    for submodule in submodules:
        logger = logging.getLogger(submodule)
        logger.setLevel(level)