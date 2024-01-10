import os
import sys

from openai import OpenAI

from cli import CLI, CLIResponse, console
from utils import Agent, Color, Model
from witch import Witch
from wizard import Wizard


class Sapphire:
    def __init__(self) -> None:
        client = self.__setup_connection()
        system = self.__detect_system()
        directory = os.getcwd()
        self.active_agent = Agent.WIZARD
        self.history = {}
        self.model = Model.THREE_FIVE_TURBO

        self.cli = CLI(
            self.history, self.get_agent, self.set_agent, self.get_model, self.set_model
        )
        self.wizard = Wizard(client, system, self.history, self.get_model)
        self.witch = Witch(client, directory, self.history)

        self.start()

    def start(self) -> None:
        while True:
            cmd = self.cli.get_user_input(self.active_agent)
            # cmd is a special command
            if cmd == CLIResponse.IGNORE:
                continue
            # reingest documents
            elif cmd == CLIResponse.REINGEST:
                self.witch.reingest()
            elif self.active_agent == Agent.WIZARD:
                self.wizard.execute_cmd(cmd)
            elif self.active_agent == Agent.WITCH:
                self.witch.answer_question(cmd)

    def __setup_connection(self) -> OpenAI:
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key is None:
            error_msg = (
                'pushpin: Make sure to add your OpenAI API KEY to your '
                'system environment as OPENAI_API_KEY. :pushpin:'
            )
            console.print(f'{Color.ERROR.value}:{error_msg}')
            sys.exit(0)
        return OpenAI(api_key=api_key)

    def __detect_system(self) -> str | None:
        platform = sys.platform
        if platform.startswith('linux'):
            return 'Linux'
        elif platform.startswith('darwin'):
            return 'MacOS'
        elif platform.startswith('win32'):
            return 'Windows'
        else:
            console.print(
                f'{Color.ERROR.value}User platform is incompatible with Sapphire :('
            )
            sys.exit(0)

    def get_agent(self) -> Agent:
        return self.active_agent

    def set_agent(self, agent: Agent) -> None:
        self.active_agent = agent

    def get_model(self) -> Model:
        return self.model

    def set_model(self, model: Model) -> None:
        self.model = model
