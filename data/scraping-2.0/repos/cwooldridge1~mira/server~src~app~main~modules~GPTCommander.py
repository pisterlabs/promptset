import openai
from .commands import *
from os import environ
from typing import Dict


openai.api_key = environ.get('OPENAI_KEY')


class GPTCommander():
    def __init__(self, ) -> None:
        self.__model = environ.get('OPENAI_COMMAND_MODEL')
        self.commands: Dict[str, Command] = {
            'ChartCommand': ChartCommand,
            'CodeCommand': CodeCommand,
            'AddTaskCommand': AddTaskCommand,
            'DeleteTaskCommand': DeleteTaskCommand,
            'FallbackCommand': FallbackCommand,
            'PlaceMarketOrderCommand': PlaceMarketOrderCommand
        }

    def getCommand(self, prompt: str) -> Command:
        """
        Uses the GPT model to extract the best fitting command and arguments which then is used to return a constructed Command instance
        """
        response: str = openai.Completion.create(
            model=self.__model,
            prompt=prompt + " ->",
        )['choices'][0]['text']

        response = response.split('\n')[0].split(',')

        command: Command = self.commands.get(response[0])
        if command is None or response[0] == 'FallbackCommand':
            return self.commands['FallbackCommand'](prompt)

        args = tuple(arg.strip() for arg in response[1:])
        print(response[0], args)

        return command(*args)
