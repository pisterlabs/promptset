#!/usr/bin/python3

# Imports
import openai


class Interaction:
    def __init__(self, chat_history):
        self.openai_api = openai.ChatCompletion()
        self.model = "gpt-3.5-turbo-16k"
        self.temperature = 0.1
        self.chat_history = chat_history

    def interact(self):
        raise NotImplementedError
