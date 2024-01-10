"""
Commit Message Generator

This module provides functionality for generating commit messages using the OpenAI API.
"""
from random import shuffle
from typing import Generator

import requests
from .config_parser import Config
from .git_diff import git_diff


class UnusedToken:
    """
    A class for managing unused tokens for API requests.
    """
    _instance = None
    constant_pool = tuple()
    access_pool = []

    def __new__(cls, string_list: list[str]):
        """
        Create a new instance of the class or return the existing instance if it already exists.
        """
        # Check if the string_list is empty, then raise an error
        if len(string_list) == 0:
            raise ValueError(
                'tokens should not be empty.\n ' +
                'Please set the environment variable GPT_TOKENS to a non-empty string.\n ' +
                '\t For example export GPT_TOKENS="token1,token2,token3"\n' +
                'Or edit the config file to set the tokens.')
        if not cls._instance:
            cls._instance = super(UnusedToken, cls).__new__(cls)
            cls._instance.constant_pool = tuple(string_list)
            cls._instance.access_pool = list(cls._instance.constant_pool)
            shuffle(cls._instance.access_pool)
        return cls._instance

    def copy_pool(self) -> None:
        """
        Copy the constant pool of tokens to the access pool.
        """
        self.access_pool = list(*self.constant_pool)

    def pop(self) -> str:
        """
        Remove and return a token from the access pool.
        If the access pool is empty, copy the pool and shuffle it.
        """
        if len(self.access_pool) == 0:
            self.copy_pool()
            shuffle(self.access_pool)
        return self.access_pool.pop(0)


def find_first_number(string: str) -> float:
    """
    Find the first number in a string.
    Split the string by space and find the first number in the list.
    """
    for word in string.split(' '):
        try:
            return float(word)
        except ValueError:
            pass
    return 0.0


def get_score(config: Config, diff: str, msg: str) -> float:
    """
    Get the score of the commit message.
    """
    unused_token = UnusedToken(config['tokens'])

    message = [
        {
            "role": "system",
            "content": config['revise']['prompt'],
        },
        {
            "role": "user",
            "content": "Source diff:\n" + diff + "\n\n" + "Commit message:\n" + msg,
        },
    ]

    data = {
        'model': config['revise']['model'],
        'messages': message,
        'max_tokens': config['revise']['max_tokens'],
        'temperature': config['revise']['temperature'],
        'top_p': config['revise']['top_p'],
        'frequency_penalty': config['revise']['frequency_penalty'],
        'presence_penalty': config['revise']['presence_penalty'],
    }

    response = requests.post(
        'https://api.openai.com/v1/chat/completions',
        headers={
            'Authorization': 'Bearer ' + unused_token.pop(),
            'Content-Type': 'application/json'
        },
        json=data,
        timeout=config['timeout'],
    )

    score = find_first_number(
        response.json()['choices'][0]['message']['content'])
    return score


def generate_commit_message(
        config: Config, repo_path: str) -> Generator[str, None, None]:
    """
    Generate commit messages using the OpenAI API.
    """
    unused_token = UnusedToken(config['tokens'])
    diff = git_diff(repo_path)

    message = [
        {
            "role": "system",
            "content": config['system_prompt'],
        },
        {
            "role": "user",
            "content": diff,
        },
    ]

    data = {
        'model': config['model'],
        'messages': message,
        'max_tokens': config['max_tokens'],
        'temperature': config['temperature'],
        'top_p': config['top_p'],
        'frequency_penalty': config['frequency_penalty'],
        'presence_penalty': config['presence_penalty'],
    }

    while True:
        #! We don't use OpenAI Python API because we don't want to couple it with our code.
        #! For hacking purposes, we use requests library to send HTTP request to OpenAI API.
        #! We need to poll the tokens to avoid rate limit.

        # Link reference: https://platform.openai.com/docs/api-reference/chat/create
        # Request from OpenAI API like:
        # curl https://api.openai.com/v1/chat/completions \
        #   -H "Content-Type: application/json" \
        #   -H "Authorization: Bearer $OPENAI_API_KEY" \
        #   -d '{
        #     "model": "gpt-3.5-turbo",
        #     "messages": [{"role": "user", "content": "Hello!"}]
        #     "max_tokens": 5,
        #     "temperature": 0.9,
        #   }'

        api_key = unused_token.pop()
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}',
        }

        response = requests.post(
            'https://api.openai.com/v1/chat/completions',
            headers=headers,
            json=data,
            timeout=config['timeout'],
        )

        # Response from OpenAI API like:
        # {
        #   "id": "scc-123",
        #   "object": "chat.completion",
        #   "created": 1619795811,
        #   "choices": [{
        #     "index": 0,
        #     "message": {
        #       "role": "assistant",
        #       "content": "\n\nHello there, how may I assist you today?",
        #     },
        #     "finish_reason": "stop"
        #   }],
        #   "usage": {
        #     "prompt_tokens": 9,
        #     "completion_tokens": 12,
        #     "total_tokens": 21
        #   }
        # }

        msg = response.json()['choices'][0]['message']['content']
        if msg == '' or config['revise']['threshold'] > get_score(
                config, diff, msg):
            # Regenerate the commit message if the score is lower than the
            # threshold
            continue
        # Yield the content of first choice
        yield msg
