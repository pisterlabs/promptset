import json
import os
from typing import List, Dict, Any, Callable

from langchain.memory import ChatMessageHistory, ConversationBufferWindowMemory
from langchain.schema import BaseMessage, messages_from_dict, messages_to_dict


def _load_messages(save_path: str) -> List[BaseMessage]:
    with open(save_path, "r") as f:
        items = json.load(f)
    messages = messages_from_dict(items)
    return messages


def _load_chat_memory(chat_memory_path: str) -> ChatMessageHistory:
    if os.path.isfile(chat_memory_path):
        messages = _load_messages(chat_memory_path)
        chat_memory = ChatMessageHistory(messages=messages)
    else:
        chat_memory = ChatMessageHistory()
    return chat_memory


class SavableWindowMemory(ConversationBufferWindowMemory):
    save_path: str
    chat_memory_file_name: str
    input_preprocessor: Callable[[Dict], Dict] = lambda x: x
    output_preprocessor: Callable[[Dict], Dict] = lambda x: x

    @classmethod
    def load(cls, save_path: str, **kwargs):
        chat_memory_file_name = kwargs.pop("chat_memory_file_name", "chat_memory.json")
        chat_memory_path = os.path.join(save_path, chat_memory_file_name)
        chat_memory = _load_chat_memory(chat_memory_path)
        return cls(
            save_path=save_path,
            chat_memory_file_name=chat_memory_file_name,
            chat_memory=chat_memory, **kwargs
        )

    @staticmethod
    def _save_messages(messages: List[BaseMessage], save_path: str):
        messages_dict = messages_to_dict(messages)
        with open(save_path, "w") as f:
            json.dump(messages_dict, f)

    def _save_chat_memory(self):
        chat_memory_path = os.path.join(self.save_path, self.chat_memory_file_name)
        messages = self.chat_memory.messages
        self._save_messages(messages, chat_memory_path)

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        super().save_context(self.input_preprocessor(inputs), self.output_preprocessor(outputs))
        self._save_chat_memory()

    def clear(self) -> None:
        super().clear()
        self._save_chat_memory()
