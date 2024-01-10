import openai
from messages import Message


class CompletionRequest:
    config: dict
    messages: list[dict]

    def __init__(self, config):
        self.config = config
        self.messages = []

    def add(self, message: Message):
        self.messages.append(message.to_dict())
        return self

    def apply(self):
        response = openai.ChatCompletion.create(
            messages=self.messages,
            **self.config
        )
        return response
