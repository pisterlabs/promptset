from typing import Optional

import re
import yaml
from langchain.chat_models.openai import ChatOpenAI
from langchain.schema import ChatMessage


class ChatroomMessage(ChatMessage):
    timestamp: int
    """UNIX timestamp of the message in seconds."""


class ChatroomConversation:
    messages: list[ChatroomMessage]
    ai_user_id: str

    def __init__(
        self,
        messages: list[ChatroomMessage],
        ai_user_id: str,
        max_tokens: Optional[int] = None,
    ) -> None:
        self.ai_user_id = ai_user_id

        self.messages = messages

        if max_tokens is not None:
            self.trim_messages_to_max_tokens(max_tokens)

        self.messages = sorted(self.messages, key=lambda msg: msg.timestamp)

    def trim_messages_to_max_tokens(
        self,
        max_tokens: int,
    ) -> set[ChatroomMessage]:
        """Trim messages to fit within the max_tokens limit, discarding the oldest messages."""
        tokens_used = 0
        messages_newest_first = sorted(self.messages, key=lambda msg: -msg.timestamp)

        self.messages = []
        for msg in messages_newest_first:
            msg_line = self.chatroom_msg_to_yaml_msg_line(msg)

            msg_line_tokens = ChatOpenAI().get_num_tokens(f'- {msg_line}')
            if tokens_used + msg_line_tokens > max_tokens:
                break

            self.messages.append(msg)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> 'ChatroomConversation':
        d = yaml.safe_load(yaml_str)

        messages = [
            cls.chatroom_msg_from_yaml_msg_line(line)
            for line in d['messages_oldest_first']
        ]

        return cls(
            messages=messages,
            ai_user_id=d['ai_user_id'],
        )

    @staticmethod
    def chatroom_msg_to_yaml_msg_line(msg: ChatroomMessage) -> str:
        return f'At {msg.timestamp}, @{msg.role} said: {msg.content}'

    @staticmethod
    def chatroom_msg_from_yaml_msg_line(line: str) -> ChatroomMessage:
        matches = re.search(r'(?s)At (\d+), @(\w+) said: (.*)', line)
        if matches:
            return ChatroomMessage(
                role=matches.group(2),
                content=matches.group(3),
                timestamp=int(matches.group(1)),
            )
        else:
            raise ValueError(f'Invalid message line: {line}')

    def to_yaml(self) -> str:
        messages_oldest_first = sorted(self.messages, key=lambda msg: msg.timestamp)
        msg_lines = [self.chatroom_msg_to_yaml_msg_line(msg) for msg in messages_oldest_first]

        return yaml.dump(
            {
                'messages_oldest_first': msg_lines,
                'ai_user_id': self.ai_user_id,
            }
        )
