"""
Load a ChatSession from an existing Twilio Conversation

The aim is to get this in `langchain.chat_loaders`
"""

from typing import Iterator
from langchain.chat_loaders.base import BaseChatLoader, ChatSession
from langchain.schema import HumanMessage, AIMessage


class TwilioConversationChatLoader(BaseChatLoader):
    def __init__(self, client, chat_service_sid, conversation_sid, ai_name="system"):
        self.client = client
        self.chat_service_sid = chat_service_sid
        self.conversation_sid = conversation_sid
        self.ai_name = ai_name

    def lazy_load(self) -> Iterator[ChatSession]:
        conversation_messages = (
            self.client.conversations.v1.services(self.chat_service_sid)
            .conversations(self.conversation_sid)
            .messages.list()
        )
        messages = []
        # TODO: What to do with Media?
        # TODO: Multiple users?
        for conv in conversation_messages:
            if conv.author == self.ai_name:
                messages.append(AIMessage(content=conv.body, additional_kwargs={
                    "date_created": conv.date_created,
                }))
            else:
                messages.append(HumanMessage(
                    content=conv.body,
                    additional_kwargs={
                        "author": conv.author,
                        "participant_sid": conv.participant_sid,
                        "date_created": conv.date_created,
                    }
                ))
        # I suppose we could page...but why?
        yield ChatSession(messages=messages) 
