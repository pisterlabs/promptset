from .Command import Command
from ....types.responses import ContentResponse
from ..OpenAI import OpenAI


class FallbackCommand(Command):
    def __init__(self, prompt):
        self.prompt = prompt

    def execute(self):
        return OpenAI.ask(self.prompt)

    def getSocketResponse(self):
        answer = self.execute()
        return ContentResponse(type='text', data={'text': f'{self.prompt}:\n {answer}'})
