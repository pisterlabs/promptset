import subprocess

from openai import OpenAI
from rich.panel import Panel
from rich.prompt import Confirm

from cli import console
from utils import Agent, Color, CommandStatus


class Wizard:
    def __init__(self, client: OpenAI, system: str, history: dict, get_model) -> None:
        self.client = client
        self.system = system
        self.get_model = get_model

        self.agent = Agent.WIZARD
        history[self.agent] = []
        self.history = history[self.agent]
        self.msgs = [
            {
                'role': 'system',
                'content': 'You are an intelligent assistant named Sapphire that helps \
                      the user execute command line commands to manage their desktop',
            }
        ]

    def execute_cmd(self, user_cmd) -> None:
        wizard_cmd = self.__get_wizard_cmd(user_cmd)
        if wizard_cmd.startswith('Invalid command'):
            error_msg = (
                f'{Color.ERROR.value}Please enter a valid command, or '
                + f'press[/] {Color.MENU.value}help;[/] {Color.ERROR.value}to '
                + 'view the help menu.'
            )
            console.print(Panel(error_msg, title=self.agent.value))
            self.history.append((user_cmd, wizard_cmd, CommandStatus.INVALID))
        elif self.__greenlight_wizard_cmd(wizard_cmd):
            subprocess.call(wizard_cmd, shell=True)
            self.history.append((user_cmd, wizard_cmd, CommandStatus.EXECUTED))
        else:
            console.print(
                Panel(
                    f'{Color.ERROR.value}:stop_sign: Aborting :stop_sign:',
                    title=self.agent.value,
                )
            )
            self.history.append((user_cmd, wizard_cmd, CommandStatus.ABORTED))

    def __greenlight_wizard_cmd(self, wizard_cmd) -> bool:
        greenlit = Confirm.ask(f':dango: Execute {Color.COMMAND.value}{wizard_cmd}')
        return greenlit

    def __get_wizard_cmd(self, user_cmd) -> str:
        cmd = (
            f'What is the script in a {self.system} environment to execute the '
            + f'following command: "{user_cmd}". Reply with just the command and '
            + 'no additional text. If it is not possible to answer with a command '
            + 'line command, reply with "Invalid command"'
        )
        self.msgs.append({'role': 'user', 'content': cmd})
        console.print(
            f'{Color.SYSTEM.value}:light_bulb: The wizard is thinking :light_bulb:'
        )
        chat = self.client.chat.completions.create(
            model=self.get_model().value, messages=self.msgs
        )
        reply = chat.choices[0].message.content
        self.msgs.append({'role': 'assistant', 'content': reply})
        return reply
