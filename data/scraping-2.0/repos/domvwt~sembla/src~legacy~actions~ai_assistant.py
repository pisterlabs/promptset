import logging

import openai

from sembla.conversation_history import ConversationHistory
from sembla.llm.openai.chat_completion import ChatCompletion

_ASSISTANT_PROMPT = """\
You are a helpful assistant that completes tasks efficiently and accurately.
You request no user interaction and function autonomously.
"""


def query_assistant(prompt: str, role: str = "user") -> str:
    messages = [
        {"role": "system", "content": _ASSISTANT_PROMPT},
        {"role": role, "content": prompt},
    ]
    conversation_history = ConversationHistory()
    conversation_history._extend_history(messages)
    chat_completion = ChatCompletion(
        model="gpt-3.5-turbo", conversation_history=conversation_history
    )
    response = chat_completion.create(
        temperature=0.2,
        n=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    logging.info("Assistant response:\n%s", response)
    return response
