from typing import Union, List

from langchain.schema import HumanMessage, AIMessage
from pydantic import BaseModel

from jonbot.backend.data_layer.models.discord_stuff.discord_message_document import DiscordMessageDocument


class ChatMemoryMessageBuffer(BaseModel):
    message_buffer: List[Union[HumanMessage, AIMessage]] = []

    @classmethod
    def from_discord_message_documents(cls,
                                       discord_message_documents: List[DiscordMessageDocument]):
        message_buffer = []
        for message in discord_message_documents:
            if message.is_bot:
                message_buffer.append(
                    AIMessage(
                        content=message.content,
                        additional_kwargs={
                            "message_id": message.message_id,
                            "type": "ai",
                        },
                    )
                )
            else:
                message_buffer.append(
                    HumanMessage(
                        content=message.content,
                        additional_kwargs={
                            "message_id": message.message_id,
                            "type": "human",
                        },
                    )
                )
        return cls(message_buffer=message_buffer)
