from typing import Any, Literal

from langchain_core.prompt_values import PromptValue
from langchain_core.prompts import ChatPromptTemplate

from pybot.prompts.base import FlexPromptValue

# TODO: this section is mainly used for formatting histroy messages.
# As Literal cannot use variables, this lead to a bit of duplication.
HUMAN_PREFIX = "<|user|>\n"
HUMAN_SUFFIX = "</s>"
AI_PREFIX = "<|assistant|>\n"
AI_SUFFIX = "</s>"


class ZephyrPromptTemplate(ChatPromptTemplate):
    """zephyr prompt template.
    See <https://huggingface.co/HuggingFaceH4/zephyr-7b-beta>
    """

    def format_prompt(self, **kwargs: Any) -> PromptValue:
        """Format prompt."""
        messages = self.format_messages(**kwargs)
        return ZephyrPromptValue(messages=messages)


class ZephyrPromptValue(FlexPromptValue):
    """Chat Markup Language prompt value."""

    system_prefix: Literal["<|system|>\n"] = "<|system|>\n"
    system_suffix: Literal["</s>"] = "</s>"
    human_prefix: Literal["<|user|>\n"] = "<|user|>\n"
    human_suffix: Literal["</s>"] = "</s>"
    ai_prefix: Literal["<|assistant|>\n"] = "<|assistant|>\n"
    ai_suffix: Literal["</s>"] = "</s>"
