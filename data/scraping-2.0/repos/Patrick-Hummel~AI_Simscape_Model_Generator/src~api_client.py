# -*- coding: utf-8 -*-

"""
Use of the "Singleton" design pattern to only allow single instances of API clients for prompt requests.

Singleton metaclass solution inspired by the answer of user "WorldSEnder" on Stack Overflow (07.07.2015):
https://stackoverflow.com/questions/31269974/why-singleton-in-python-calls-init-multiple-times-and-how-to-avoid-it
Answer: https://stackoverflow.com/a/31270973 User: https://stackoverflow.com/users/3102935/worldsender


Last modification: 28.11.2023
"""

__version__ = "1"
__author__ = "Patrick Hummel"

import json

from openai import OpenAI

from config.gobal_constants import PATH_DEFAULT_ABSTRACT_SYSTEM_JSON_SCHEMA_FILE


class Singleton(type):
    def __init__(self, name, bases, mmbs):
        super(Singleton, self).__init__(name, bases, mmbs)
        self._instance = super(Singleton, self).__call__()

    def __call__(self, *args, **kw):
        return self._instance


class OpenAIGPTClient(metaclass=Singleton):

    def __init__(self):
        self.client = OpenAI()
        print("Newly created!")

        # Load JSON schema and prepare to be added to function call parameter
        with open(PATH_DEFAULT_ABSTRACT_SYSTEM_JSON_SCHEMA_FILE, 'r') as file:
            self.json_response_schema = json.load(file)

        self.json_response_schema.pop('$schema', None)

    def request(self, prompt: str) -> str:

        # Request a valid json as response format
        completion = self.client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system",
                 "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )

        print(f"Tokens (Response): Prompt = {completion.usage.prompt_tokens}, "
              f"Completion = {completion.usage.completion_tokens}, "
              f"Total = {completion.usage.total_tokens}")

        return completion.choices[0].message.content

    def request_as_function_call(self, prompt: str) -> str:

        # Request a valid json as response format
        completion = self.client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            response_format={"type": "json_object"},
            messages=[
                {"role": "user", "content": prompt}
            ],
            functions=[
                {
                    "name": "createElectricalCircuitObject",
                    "parameters": self.json_response_schema
                }
            ],
            function_call={"name": "createElectricalCircuitObject"}
        )

        print(f"Tokens (Response): Prompt = {completion.usage.prompt_tokens}, "
              f"Completion = {completion.usage.completion_tokens}, "
              f"Total = {completion.usage.total_tokens}")

        return completion.choices[0].message.function_call.arguments


class GoogleBardClient(metaclass=Singleton):

    def __init__(self):
        pass

    def request(self, prompt: str):
        raise NotImplementedError
