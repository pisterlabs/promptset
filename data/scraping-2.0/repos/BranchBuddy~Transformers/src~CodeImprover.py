from src.OpenaiHandler import OpenAiHandler


class CodeImprover:
    def __init__(self) -> None:
        self.handler = OpenAiHandler()

    def improveCode(self, code: str):
        self.handler.createAssistant('improve')
        response = self.handler.handleTask('improve', code)
        return response
    
    def commentCode(self, code: str):
        self.handler.createAssistant('comments')
        response = self.handler.handleTask('comments', code)
        return response