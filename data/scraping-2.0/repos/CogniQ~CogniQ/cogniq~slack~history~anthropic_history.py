from __future__ import annotations
from typing import *

import logging

logger = logging.getLogger(__name__)

from .openai_history import OpenAIHistory
from cogniq.openai import system_message, user_message, assistant_message


class AnthropicHistory(OpenAIHistory):
    def _convert_to_chat_sequence(self, *, messages, bot_user_id):
        chat_sequence = ""
        for message in messages:
            if message.get("user") == bot_user_id:
                chat_sequence += f"\n\nAssistant: {message.get('text')}"
            else:
                chat_sequence += f"\n\nHuman: {message.get('text')}"
            if message.get("replies"):
                for reply in message.get("replies"):
                    if reply.get("user") == bot_user_id:
                        chat_sequence += f"\n\nAssistant: {reply.get('text')}"
                    else:
                        chat_sequence += f"\n\nHuman: {reply.get('text')}"
        return chat_sequence

    def openai_to_anthropic(self, *, message_history):
        # Initialize an empty list to store the messages
        messages = []

        # Iterate over the chat history
        for message in message_history:
            # Extract the role and content from the message
            role = message["role"]
            content = message["content"]

            # Add the message to the list
            if role == "user":
                messages.append(f"Human: {content}")
            elif role == "assistant":
                messages.append(f"Assistant: {content}")
            elif role == "system":
                messages.append(f"Human: {content}")

        # Convert the list of messages to the Anthropic format
        anthropic_chat_history = "\n\n".join(messages)
        return anthropic_chat_history
