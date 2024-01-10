from .Command import Command
from ....types.responses import ContentResponse
from ..OpenAI import OpenAI


class CodeCommand(Command):
    def __init__(self, language, *args):
        self.language = language
        self.args = args

    def execute(self):
        return OpenAI.ask(f"Write code in {self.language} that {' '.join(self.args)}")

    def getSocketResponse(self) -> ContentResponse:
        code = self.execute()
        return ContentResponse(type='code', data={'text': code, 'language': self.language})
