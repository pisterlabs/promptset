import logging

import openai

from conversation import Conversation

logging.basicConfig(level=logging.DEBUG)


class Executor:
    def execute():
        raise NotImplementedError


class OpenAI(Executor):
    def __init__(self, secret, org, model) -> None:
        openai.api_key = secret
        openai.organization = org
        self.model = model

    def execute(self, conversation: Conversation, temp: float = 0.7) -> str:
        """Sends the conversation for completion using OpenAI's API

        Args:
            conversation (Conversation): Conversation to be completed

        Raises:
            NotImplementedError: _description_

        Returns:
            str: Result of the conversation completion
        """
        logging.debug(self.extract_prompt(conversation))

        return "TEST"
        # return openai.Completion.create(
        #    engine=self.model,
        #    prompt=self.extract_prompt(conversation),
        #    temperature=temp,
        # )

    def extract_prompt(self, conversation):
        chat = []
        for message in conversation.messages:
            chat.append({message["role"]: message["content"]})
        return chat


class LocalModel(Executor):
    pass
