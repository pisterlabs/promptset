from src.OpenaiHandler import OpenAiHandler


class CodeSummarizer:

    def __init__(self):
        self.handler = OpenAiHandler()

    def summarizeCode(self, code: str):
        # This code is for v1 of the openai package: pypi.org/project/openai
        self.handler.createAssistant('summary')
        response = self.handler.handleTask('summary',code)
        return response
