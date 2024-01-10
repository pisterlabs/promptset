from dataclasses import dataclass

import openai
from argklass.command import Command

from milainference.core.client import init_client


class Client(Command):
    """Send an inference request to a server"""

    name: str = "client"

    @dataclass
    class Arguments:
        prompt: str  # Prompt
        model: str = None  # Model Name
        short: bool = False  # Only print the result and nothing else

    def execute(self, args):
        model = init_client(args.model)

        completion = openai.Completion.create(model=model, prompt=args.prompt)

        if args.short:
            for choices in completion["choices"]:
                print(choices["text"])
        else:
            print(completion)


COMMANDS = Client
