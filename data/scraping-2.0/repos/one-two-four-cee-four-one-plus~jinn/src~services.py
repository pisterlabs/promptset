import json
import inspect
import traceback
import tempfile
import logging
import textwrap

from openai import OpenAI

from constants import OPENAI_FUNCTION_SCHEMA, CRAFT_INCANTATION_SCHEMA
from utils import unwrap_content, define_function, NoDefaults
from config import LOG_PATH


logger = logging.getLogger('jinn_openai')
logger.setLevel(logging.INFO)
handler = logging.FileHandler(LOG_PATH)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def describe_function(key, model, code):
    response = OpenAI(api_key=key).chat.completions.create(
        model=model,
        temperature=0,
        messages=[{
            "role": "user",
            "content": (
                f"Describe this python function: {code} in this schema {OPENAI_FUNCTION_SCHEMA}. "
                "I want only json in response, nothing else."
            )
        }]
    )
    ret = unwrap_content(response.choices[0].message.content, 'json')
    logger.info(f'describe_function({code}) = {ret}')
    return ret


def craft_incantation(key, model, retries, text):
    messages = [{
        "role": "user",
        "content": (
            "Write a python function according to the request. "
            "I want only python code in response, nothing else. "
            "Don't comment the code, don't ask for user's input, don't print anything and "
            "don't do any exception handling unless it's necessary. "
            "Don't use keywords, variadic arguments. "
            "Dont use any types other than numbers, strings and booleans as "
            "function's arguments. "
            "Don't use lists, tuples, dictionaries, sets, etc. "
            "Don't use any global variables. "
            "Try not to use any external libraries, only built-in ones. "
            "Function's name should be as descriptive as possible. "
            "If you're certain that you need to use some external package, "
            "and that package is not a part of python's standard library, "
            "import it in the beginning of the function like this:\n"
            "try:\n"
            "    import requests\n"
            "except ImportError:\n"
            "    import pip\n"
            "    pip.main(['install', 'requests'])\n"
            "If function needs any parameters (like credentials,"
            " configuration, etc), they should be passed as arguments."
            f" \nRequest:\n{text}"
        )
    }]
    response = OpenAI(api_key=key).chat.completions.create(
        model=model,
        temperature=0,
        messages=messages,
    )
    messages.append({"role": "assistant", "content": response.choices[0].message.content})

    last_e = None
    for i in range(int(retries)):
        code = unwrap_content(response.choices[0].message.content, 'python')
        code = NoDefaults.in_(code)
        try:
            name, _ = define_function(code)
            logger.info(f'craft_incantation({text}) = {name}, {code}')
            return name, code
        except Exception as e:
            last_e = e
            messages.append({
                "role": "assistant",
                "content": (
                    f"{''.join(traceback.format_exception(e))}\n"
                    "Fix this error. I want only python code in response, nothing else."
                )
            })
            continue
    else:
        return last_e


def wish(key, model, text, incantations, allow_craft=False, call=True):
    tools = [value['schema'] for value in incantations.values()]
    if allow_craft:
        tools.append(CRAFT_INCANTATION_SCHEMA)
        instructions = (
            'Use one of the provided externals tools to fulfill the request.'
            ' If there is no suitable tool available, define one instead of fulfilling'
            ' the original request. The tool should be generic enough to be useful in'
            ' other situations, but should only use numbers, strings and booleans as types.'
            ' If the request has a word "question" in the beginning, just answer it shortly.'
            ' When calling tools with credentials, configuration, etc, that cannot'
            ' be infered from the request, pass empty strings instead of them.'
            f'\nRequest:\n{text}'
        )
    else:
        instructions = (
            'Use one of the provided externals tools to fulfill the request. '
            ' If the request has a word "question" in the beginning, just answer it shortly.'
            f'\nRequest:\n{text}'
        )
    response = OpenAI(api_key=key).chat.completions.create(
        model=model,
        temperature=0,
        messages=[{
            "role": "user",
            "content": instructions
        }],
        tools=tools,
        tool_choice="auto"
    )
    response_message = response.choices[0].message
    if tool_calls := response_message.tool_calls:
        if tool_calls[0].function.name == 'craft_incantation':
            logger.info(f'wish({text}) = craft_incantation{tool_calls[0].function.arguments}')
            return 'craft_incantation', tool_calls[0].function.arguments
        elif call:
            try:
                incantation = incantations[tool_calls[0].function.name]
                args = json.loads(tool_calls[0].function.arguments)
                logger.info(f'wish({text}) = {tool_calls[0].function.name} {args}')
                return incantation['object'].execute({'args': args})['result']
            except Exception as e:
                return incantation['object'], tool_calls[0].function.arguments, e
        else:
            logger.info(f'wish({text}) = {tool_calls[0].function.name}{tool_calls[0].function.arguments}')
            return incantations[tool_calls[0].function.name], tool_calls[0].function.arguments
    else:
        logger.info(f'wish({text}) = {response_message.content}')
        return response_message.content


def adjust(key, model, code, reason):
    instructions = (
        "I need you to adjust the python function according to the request. "
        "I will provide it's code and the reason why it needs to be adjusted. "
        "You shouldn't change the function's name, arguments, return type, "
        " and you shouldn't add any new arguments. "
        "It should serve the same purpose as the original function. "
        "I want only python code in response, nothing else."
        f" Code:\n{code}\nReason:\n{reason}"
    )
    response = OpenAI(api_key=key).chat.completions.create(
        model=model,
        temperature=0,
        messages=[{"role": "user", "content": instructions}],
    )
    code = unwrap_content(response.choices[0].message.content, 'python')
    try:
        name, _ = define_function(code)
        logger.info(f'adjust({name}, {reason}) = {code}')
        return code
    except Exception as e:
        return e


def fix(key, model, code, request, traceback):
    instructions = (
        "I need you to fix python function. I will provide it's code, call"
        " arguments formatted in some json schema and error traceback. Fix"
        " this error. I want only python code in response, nothing else."
        f" Code:\n{code}\nArguments:\n{request}\n"
        f"Traceback:\n{traceback}"
    )
    _, func = define_function(code)
    arguments = set(inspect.getargspec(func).args)
    arguments = {k: v for k, v in json.loads(request).items() if k in arguments}
    arguments = json.dumps(arguments)
    response = OpenAI(api_key=key).chat.completions.create(
        model=model,
        temperature=0,
        messages=[{"role": "user", "content": instructions}],
    )
    code = unwrap_content(response.choices[0].message.content, 'python')
    try:
        define_function(code)
        logger.info(f'fix({code}, {request}, {traceback}) = {code}')
        return code
    except Exception as e:
        return e


def stt(key, data):
    with tempfile.NamedTemporaryFile(suffix='.m4a') as temp:
        with open(temp.name, 'wb') as f:
            f.write(data)
        with open(temp.name, 'rb') as temp_read:
            response = OpenAI(api_key=key).audio.transcriptions.create(
                model="whisper-1",
                file=temp_read,
            )
        logger.info(f'stt() = {response.text}')
        return response.text


def tts(key, text):
    response = OpenAI(api_key=key).audio.speech.create(
        model="tts-1-hd",
        voice="nova",
        input=text,
    )
    return response.content
