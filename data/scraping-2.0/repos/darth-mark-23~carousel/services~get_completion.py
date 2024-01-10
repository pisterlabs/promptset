import openai
import os

from models.message import Message

openai.api_key = os.getenv('OPENAI_API_KEY')
max_tokens = 2000
temperature = 0.7

def get_completion(model: str, function_definitions: list[str], messages: list[Message]) -> any:
    if (model is None):
        raise Exception('Completion requested with missing model parameter')

    if (function_definitions is None):
            raise Exception('Completion requested with missing functions parameter')
    
    if (messages is None):
        raise Exception('Completion requested with blank messages parameter')  

    messages_json = [message.to_json() for message in messages]

    args = {
        'model': model,
        'messages': messages_json,
        'max_tokens': max_tokens,
        'temperature': temperature
    }

    if function_definitions:
        args['functions'] = function_definitions

    completion = openai.ChatCompletion.create(**args)
    
    return completion
