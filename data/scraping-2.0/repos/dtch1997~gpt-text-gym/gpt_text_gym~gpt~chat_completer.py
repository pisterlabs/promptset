""" Interface to GPT model."""

import openai
import dotenv

from ml_collections import config_dict
from dataclasses import dataclass
from gpt_text_gym import ROOT_DIR
from typing import List, NewType, Dict, Optional
from gpt_text_gym.gpt.message import Message, RawMessage, default_system_message
from gpt_text_gym.gpt.utils import remove_leading_whitespace


def get_chatgpt_system_message():
    content = """
        You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture.
        Knowledge cutoff: 2021-09
        Current date: 2023-06-26
    """
    return Message(role="system", content=remove_leading_whitespace(content))


def openai_chat_completion_create(
    model: str,
    messages: List[RawMessage],
    n: int,
    temperature: float,
    max_tokens: Optional[int],
    **kwargs,
):
    """Wrapper around OpenAI's ChatCompletion.create method."""
    return openai.ChatCompletion.create(
        model=model,
        messages=messages,
        n=n,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )


class GPTChatCompleter:
    def __init__(
        self,
        model: str = "gpt-4",
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        n: int = 1,
    ):
        openai.api_key = dotenv.get_key(ROOT_DIR / ".env", "API_KEY")
        self.chat_history: List[Message] = []
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.n = n

    def clear(self):
        self.chat_history = []

    def generate_chat_completion(self, **kwargs):
        messages = [message.to_dict() for message in self.chat_history]
        response = openai_chat_completion_create(
            model=self.model,
            messages=messages,
            n=self.n,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **kwargs,
        )

        choice = response["choices"][0]
        msg: Message = Message.from_dict(choice["message"])
        return msg

    def add_message(self, message: Message):
        self.chat_history.append(message)


if __name__ == "__main__":

    chatbot = GPTChatCompleter()
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "Translate the following English text to French: 'Hello, how are you?'",
        },
    ]
    messages = [Message.from_dict(message) for message in messages]
    for message in messages:
        print(message)
        chatbot.add_message(message)
    reply = chatbot.generate_chat_completion(messages)
    print(reply)
