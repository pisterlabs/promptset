from typing import List

from langchain.schema import AIMessage

from .base import ChatLLM


class MockLLM(ChatLLM):
    def __init__(self, responses: List[str]):
        self.responses = responses
        self.counter = 0
    
    def __call__(self, messages):
        response = self.responses[self.counter % len(self.responses)]
        self.counter += 1
        return AIMessage(content=response)