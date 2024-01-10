from abc import ABC, abstractmethod
from typing import ClassVar
from enum import Enum
from dataclasses import dataclass

import openai
from openai.openai_object import OpenAIObject

from django.conf import settings


class ChatRole(Enum):
    USER = 'user'
    SYSTEM = 'system'
    ASSISTANT = 'assistant'


@dataclass
class ChatMessage:
    content: str
    role: ChatRole = ChatRole.ASSISTANT

    def as_dict(self) -> dict:
        return {
            'content': self.content,
            'role': self.role.value
        }


class IChatGPT(ABC):

    @abstractmethod
    def send(self, message: str) -> ChatMessage:
        pass

    @abstractmethod
    def send_chat(self, chat_message: list[ChatMessage]) -> ChatMessage:
        pass


class ChatGPTAPI(IChatGPT):

    model: ClassVar[str] = "gpt-3.5-turbo"

    def __init__(self):
        self.api_key = settings.OPENAI_API_KEY
        openai.api_key = self.api_key
        self.chat = openai.ChatCompletion

    def send(self, chat_message: ChatMessage) -> ChatMessage:
        """Gets chat_message with prompt and return the response from assistance"""
        messages = [chat_message.as_dict()]
        response = self.__get_api_response(messages)
        return self.__get_chat_message_from(response)

    def send_chat(self, chat_messages: list[ChatMessage]) -> ChatMessage:
        """Gets chat list and return the response from assistance"""
        messages = [message.as_dict() for message in chat_messages]
        response = self.__get_response(messages)
        return self.__get_chat_message_from(response)

    def __get_api_response(self, messages: list[ChatMessage]) -> OpenAIObject:
        response = self.chat.create(
            model=self.model,
            messages=messages
        )
        return response

    def __get_chat_message_from(self, response: OpenAIObject) -> ChatMessage:
        content = response.choices[0].message.content
        role = response.choices[0].message.role
        return ChatMessage(content, role)
