'''
    Wrapper around openai functions, to handle :
    - prompt saving,
    - buffering prompts,
    - chat conversation handling in cyclic buffer,
'''
import logging
import os
import openai

from helpers.json import jsonWrite

# Prompt global counter
__prompt_counter = 0
# Response global counter
__response_counter = 0

# Get Open AI key from env
openai.api_key = os.environ.get('OPENAI_API_KEY', None)
if (openai.api_key is None):
    raise ValueError('Open AI key not found in env.')


def MessagesToPrompt(messages: list):
    ''' Convert messages to prompt string.'''
    prompt = ''
    for message in messages:
        prompt += f'{message["role"]}: {message["content"]}\n'

    return prompt


def GptMessagesSave(messages: list) -> None:
    ''' Save messages as prompt to temp/prompt{counter}.txt.'''
    global __prompt_counter

    if (messages is None) or (len(messages) == 0):
        logging.error('(GptChat) Messages list is empty!')
        return

    with open(f'temp/prompt{__prompt_counter}.txt', 'w') as f:
        f.write(MessagesToPrompt(messages))
        __prompt_counter += 1


def GptResponseSave(response: dict) -> None:
    ''' Save response to temp/response{counter}.txt.'''
    global __response_counter

    if (response is None) or ('choices' not in response) or (len(response['choices']) == 0):
        logging.error('(GptChat) Invalid response!')
        return

    jsonWrite(
        filename=f'temp/response{__response_counter}.json', data=response)
    __response_counter += 1


def GptPrompt(messages: list,
              model: str = 'gpt-3.5-turbo',
              temperature: float = 0.7):
    ''' Simple wrapper prompt GPT with messages.'''
    # Save prompt message to file
    GptMessagesSave(messages)

    # Send request to Open AI
    response = openai.ChatCompletion.create(model=model,
                                            messages=messages,
                                            temperature=temperature)

    # Save response to file
    GptResponseSave(response)

    return response
