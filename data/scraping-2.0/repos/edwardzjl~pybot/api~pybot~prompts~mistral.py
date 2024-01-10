from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.prompt_values import PromptValue
from langchain_core.prompts import ChatPromptTemplate

from pybot.prompts.base import FlexPromptValue

# TODO: this section is mainly used for formatting histroy messages.
# As Literal cannot use variables, this lead to a bit of duplication.
HUMAN_PREFIX = "[INST] "
HUMAN_SUFFIX = " [/INST]"
AI_PREFIX = " "
AI_SUFFIX = "</s>"


class MistralPromptTemplate(ChatPromptTemplate):
    """Mistral prompt template.
    Mistral has a very special prompt format similar to Llama 2,
    in which the system message is 'squashed' into the first user message.
    See <https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1#instruction-format>
    """

    def format_prompt(self, **kwargs: Any) -> PromptValue:
        """Format prompt."""
        messages = self.format_messages(**kwargs)
        return MistralPromptValue(messages=messages)


class MistralPromptValue(FlexPromptValue):
    human_prefix: Literal[" [INST] "] = " [INST] "
    human_suffix: Literal[" [/INST]"] = " [/INST]"
    ai_prefix: Literal[" "] = " "
    ai_suffix: Literal["</s>"] = "</s>"

    def to_string(self) -> str:
        """Return prompt as string."""
        if not isinstance(self.messages[0], SystemMessage):
            msgs = self.messages
        else:
            msgs = [
                BaseMessage(
                    type=self.messages[1].type,
                    content=f"{self.messages[0].content} {self.messages[1].content}",
                )
            ] + self.messages[2:]

        string_messages = []
        for m in msgs:
            prefix = self.get_prefix(m)
            suffix = self.get_suffix(m)
            message = f"{prefix}{m.content}{suffix}"
            if isinstance(m, AIMessage) and "function_call" in m.additional_kwargs:
                message += f"{m.additional_kwargs['function_call']}"
            string_messages.append(message)
        msgs = "".join(string_messages)
        return "<s>" + msgs
