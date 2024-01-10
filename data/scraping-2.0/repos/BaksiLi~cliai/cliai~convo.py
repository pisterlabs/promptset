#!/usr/bin/env python # -*- coding: utf-8 -*-

from typing import Dict, List

import openai
from cliai.util import print_not_implemented
from openai.openai_object import OpenAIObject


class MessageList(List[Dict[str, str]]):

    def __init__(self):
        super().__init__()

    def __repr__(self):
        return f"MessageList({super().__repr__()})"

    def __str__(self):
        return self.__repr__()

    def append(self, item: Dict[str, str]):
        if not isinstance(item, dict):
            raise TypeError("Item must be a dictionary.")
        super().append(item)

    def update_system(self, content: str):
        """
        Only the latest system prompt will be effective (experiment result).
        """
        # remove all messages with "role": "system"
        self[:] = [msg for msg in self if msg.get("role") != "system"]

        # append the new
        self.append({'role': 'system', 'content': content})

    def user_says(self, content: str):
        super().append({'role': 'user', 'content': f'{content}'})

    def assistant_says(self, content: str):
        super().append({'role': 'assistant', 'content': f'{content}'})

    def recall_last(self):
        super().pop()


class Conversation(MessageList):
    """
    This is similar to class OpenAIObject, but it does more.
    """
    def __init__(self):
        self.model = model
        self.id = id_
        self.created = created
        self.usage = usage

        self.num_choices: int = 1
        self.temperature: float = 2  # [0,2]
        # self.nucleus_sampling = top_p
        # self.pres_penalty = 
        # self.freq_penalty = 
        # self.logit_bias: Dict = 
        # self.max_tokens: int
        # self.user: Optional[str] = hash(user)

    def __str__(self):
        # index
        return 

    def show(self):
        # only show head and the end of the conversation
        pass

    def save(self, path: str):
        """
        Export the conversation to a file.
        """
        pass


def make_request(messages: MessageList) -> OpenAIObject:
    response = openai.ChatCompletion.create(model='gpt-3.5-turbo',
                                            messages=messages)
    return response


def save_convo(messages: MessageList) -> None:
    print_not_implemented()
    pass


def load_convo():
    # print(Fore.RED + 'This function is not available by far!')
    pass
