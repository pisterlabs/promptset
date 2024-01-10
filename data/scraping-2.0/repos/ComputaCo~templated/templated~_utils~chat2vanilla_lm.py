from functools import wraps
from typing import Type

from langchain.schema import SystemMessage


def Chat2VanillaLM(ChatModel: Type):
    """
    Converts a given chat model class into a class that can be used as a language model.

    Args:
        ChatModel (Type[langchain.chat_models.base.BaseChatModel]): A class representing a chat model.

    Returns:
        Type: A new class representing a language model.

    Example:
        >>> from langchain.chat_models import ChatOpenAI
        >>> from templated._utils.chat2vanilla_lm import Chat2VanillaLM
        >>> LLMChatOpenAI = Chat2VanillaLM(ChatOpenAI)
        >>> llm = LLMChatOpenAI()
        >>> type(llm("Hello, world!")) == str
        True
    """
    old_call = ChatModel.__call__

    @wraps(ChatModel.__call__)
    def __call__(self, prompt: str, *args, **kwargs):
        return old_call(self, [SystemMessage(content=prompt)], *args, **kwargs).content

    return type(ChatModel.__name__, (ChatModel,), {"__call__": __call__})
