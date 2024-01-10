from typing import Any, Literal

from langchain_core.prompt_values import PromptValue
from langchain_core.prompts import ChatPromptTemplate

from pybot.prompts.base import FlexPromptValue

# TODO: this section is mainly used for formatting histroy messages.
# As Literal cannot use variables, this lead to a bit of duplication.
HUMAN_PREFIX = "<|im_start|>user\n"
HUMAN_SUFFIX = "<|im_end|>"
AI_PREFIX = "<|im_start|>assistant\n"
AI_SUFFIX = "<|im_end|>"


class ChatMLPromptTemplate(ChatPromptTemplate):
    """A prompt template for Chat Markup Language models.
    See <https://github.com/openai/openai-python/blob/main/chatml.md>
    """

    def format_prompt(self, **kwargs: Any) -> PromptValue:
        """Format prompt."""
        messages = self.format_messages(**kwargs)
        return ChatMLPromptValue(messages=messages)


class ChatMLPromptValue(FlexPromptValue):
    """Chat Markup Language prompt value."""

    system_prefix: Literal["<|im_start|>system\n"] = "<|im_start|>system\n"
    system_suffix: Literal["<|im_end|>"] = "<|im_end|>"
    human_prefix: Literal["<|im_start|>user\n"] = "<|im_start|>user\n"
    human_suffix: Literal["<|im_end|>"] = "<|im_end|>"
    ai_prefix: Literal["<|im_start|>assistant\n"] = "<|im_start|>assistant\n"
    ai_suffix: Literal["<|im_end|>"] = "<|im_end|>"
