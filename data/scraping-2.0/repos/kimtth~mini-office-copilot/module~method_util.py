
import logging
import os
import openai
import re
from module.odsl_interpreter import generate_odsl_execute
from typing import List
from dotenv import load_dotenv
from module.enum_type import Speaker
from module.prompt_mixer import return_prompt

load_dotenv(verbose=False)
openai.api_type = "azure"
openai.api_base = os.getenv("AZURE_OPEN_AI_ENDPOINT")
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
openai.api_version = os.getenv("AZURE_OPENAI_API_VERSION_CHAT")


def chat_completion(conversation_history: List, question: str, prompt_type: str) -> str:
    """
    Generates a chat completion response using OpenAI's GPT-3 model.

    Args:
        conversation_history (List): A list of previous conversation messages.
        question (str): The user's question to generate a response for.
        prompt_type (str): The type of prompt to use for generating the response.

    Returns:
        str: The generated chat completion response.
    """

    message_history = [{"role": Speaker.SYSTEM.value, "content": return_prompt(
        prompt_type)}] + conversation_history + [{"role": Speaker.USER.value, "content": question}]

    try:
        response = openai.ChatCompletion.create(
            engine=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt35"),
            messages=message_history,
            temperature=0.7,
            max_tokens=800,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None)

        logging.info('<chat_completion>')
        msg = response["choices"][0]["message"]["content"]
        logging.info(msg)
        return msg
    except Exception as e:
        print(e)
        raise Exception('Failed to generate chat completion')


def try_parse_int(str_input: str) -> bool:
    try:
        int(str_input)
        return True
    except ValueError:
        return False


def try_parse_odsl(str_input: str) -> bool:
    try:
        # check str_iput has function name
        func_list = get_func_list()
        
        if any(func in str_input for func in func_list):
            generate_odsl_execute(str_input)
            return True
        else:
            return False
    except Exception as e:
        logging.info(f'<try_parse_odsl>:{e}')
        return False


def try_get_first_parameter_in_function_call(func_call: str) -> str:
    # Find the part of the string that is within parentheses
    match = re.search('\((.*?)\)', func_call)
    if match:
        # Split the string within parentheses by comma
        params = match.group(1).split(',')
        # Return the first parameter after stripping leading/trailing white spaces
        return params[0].strip()


def replace_first_param(function_call, new_param) -> str:
    # Find the part of the string that is within parentheses
    match = re.search('\((.*?)\)', function_call)
    if match:
        # Split the string within parentheses by comma
        params = match.group(1).split(',')
        # Replace the first parameter with the new parameter
        params[0] = f'\"{new_param}\"'
        # Join the parameters back into a string with commas
        new_params = ','.join(params)
        # Replace the old parameters with the new parameters in the function call
        return re.sub('\((.*?)\)', '(' + new_params + ')', function_call)


def get_func_list() -> List:
    func_list = ['add_outlook_schedule', 'modify_outlook_schedule', 'remove_outlook_schedule']
    return func_list

def get_func_list_with_schedule_id() -> List:
    func_list = ['modify_outlook_schedule', 'remove_outlook_schedule']
    return func_list


def get_func_list_without_schedule_id() -> List:
    func_list = ['add_outlook_schedule']
    return func_list
