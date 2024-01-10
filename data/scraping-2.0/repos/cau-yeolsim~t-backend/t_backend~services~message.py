from threading import Thread
from typing import Type

from langchain.schema.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
)

from t_backend.cipher import aes_encrypt
from t_backend.models import Message
from t_backend.repositories.message import MessageRepository
from t_backend.repositories.openai import OpenAIRepository


class MessageService:
    def __init__(
        self, message_repository: MessageRepository, openai_repository: OpenAIRepository
    ):
        # gpt 관련 부분 추가
        self._message_repository = message_repository
        self._openai_repository = openai_repository

    def create_message(self, chat_id: int, content: str) -> Message:
        # user_message
        self._message_repository.create_message(
            chat_id=chat_id, content=content, is_user=True
        )
        # bot message
        message = self._message_repository.create_message(
            chat_id=chat_id, content="", is_user=False
        )
        thread = Thread(
            target=self.send_message,
            args=(message.id, self._get_messages(chat_id)),
        )
        thread.start()
        return message

    def send_message(self, message_id: int, messages: list[BaseMessage]):
        content = self._openai_repository.send_message(
            message_id=message_id, messages=messages
        )
        self._message_repository.update_is_complete_true(
            message_id=message_id, content=content
        )

    def _get_messages(self, chat_id: int) -> list[BaseMessage]:
        message_list = self.get_message_list(chat_id=chat_id)
        return [
            AIMessage(content=message.content)
            if message.created_by == "TIRO"
            else HumanMessage(content=message.content)
            for message in message_list
        ]

    def get_message_list(self, chat_id: int) -> list[Type[Message]]:
        messages = self._message_repository.get_messages(chat_id=chat_id)
        for message in messages:
            if not message.is_complete:
                message.encrypted_content = aes_encrypt(
                    self._message_repository.get_cached_message_content(message.id)
                )
        return messages

    def get_message(self, message_id: int) -> Type[Message] | None:
        return self._message_repository.get_message(message_id=message_id)
