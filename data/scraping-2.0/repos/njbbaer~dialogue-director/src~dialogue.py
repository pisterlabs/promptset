import sys
import readline

from src.context import Context
from src.config import Config
from src.llm import OpenAI
from src.message import Message


class Dialogue:
    def __init__(self, config_filepath):
        self.config = Config(config_filepath)
        self.context = Context(self.config)
        self.llm = OpenAI(self.config)

    def loop(self):
        paused_last = True
        while True:
            for name in self.config.agents:
                if self.config.get_message_type(name) == 'manual':
                    self._input_speech(name)
                    paused_last = True
                else:
                    if not paused_last:
                        self._pause()
                    self._generate_speech(name)
                    paused_last = False

    def _generate_speech(self, name):
        rendered_messages = self._render_messages(name)
        response = self.llm.complete(rendered_messages)
        message_type = self.config.get_message_type(name)
        message = Message.create(name, response, message_type)
        self.context.append_message(message)
        message.print()

    def _input_speech(self, name):
        response = input(f"{name}: ")
        print()
        self._reload_files()
        if response:
            message = Message(name, response)
            self.context.append_message(message)

    def _render_messages(self, name):
        rendered_messages = [{'role': 'system', 'content': self.config.get_prompt(name)}]
        for message in self.context.messages:
            rendered_message = message.render_message(name)
            rendered_messages.append(rendered_message)
        return rendered_messages

    def _pause(self):
        input("Press enter to continue...")
        sys.stdout.write("\033[F")  # Move the cursor up one line
        sys.stdout.write("\033[K")  # Clear the current line
        self._reload_files()

    def _reload_files(self):
        self.context.reload()
        self.config.reload()
