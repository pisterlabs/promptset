from typing import Any, Literal

from langchain_core.prompt_values import PromptValue
from langchain_core.prompts import ChatPromptTemplate

from pybot.prompts.base import FlexPromptValue

# TODO: this section is mainly used for formatting histroy messages.
# As Literal cannot use variables, this lead to a bit of duplication.
HUMAN_PREFIX = "User: "
HUMAN_SUFFIX = "\n"
AI_PREFIX = "ASSISTANT: "
AI_SUFFIX = "\n"


class VicunaPromptTemplate(ChatPromptTemplate):
    """Vicuna prompt template.
    See <https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py>
    """

    def format_prompt(self, **kwargs: Any) -> PromptValue:
        """Format prompt."""
        messages = self.format_messages(**kwargs)
        return VicunaPromptValue(messages=messages)


class VicunaPromptValue(FlexPromptValue):
    """Vicuna prompt value."""

    human_prefix: Literal["USER: "] = "USER: "
    ai_prefix: Literal["ASSISTANT: "] = "ASSISTANT: "

    def to_string(self) -> str:
        seq = super().to_string()
        return seq[:-1]
