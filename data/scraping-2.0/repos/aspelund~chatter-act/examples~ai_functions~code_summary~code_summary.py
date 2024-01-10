import os
from utils.Config import Config
from openai_api import openai_api


def code_summary(path):
    if path == None or path == '.':
        path = ''
    if path.startswith('./'):
        path = path[2:]

    if path.startswith('.') or path.startswith('..') or path.startswith('/') or path.startswith('~'):
        return {'role': 'system', 'content': 'Invalid path.'}

    full_path = os.path.join(Config.project_path, path)
    if os.path.isfile(full_path):
        with open(full_path, 'r') as file:
            file_content = file.read()
        system_instruction = """Given the file in the user message:
Please mention the dependencies of the following script that are from within the same project.
For each function, provide a signature including the name, input types, and output type.
Please describe the format of the returned dictionaries for the functions.
Exclude any external libraries such as 'os', 'shutil', etc., from the dependencies.
Do not provide any descriptive text or summaries of the function's purpose.
Only consider the types of inputs and outputs, not their actual values or behavior."""
        system_message = {'role': 'system', 'content': system_instruction}
        user_message = {'role': 'user', 'content': file_content}
        messages = [system_message, user_message]
        response = openai_api(messages)
        return_message = {'role': 'function',
                          'name': 'file_summary', 'content': response["content"]}
        return return_message
    else:
        return {'role': 'system', 'content': 'The path does not exist.'}
