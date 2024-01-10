from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.prompt_values import PromptValue
from langchain_core.prompts import ChatPromptTemplate

from pybot.prompts.base import FlexPromptValue

# TODO: this section is mainly used for formatting histroy messages.
# As Literal cannot use variables, this lead to a bit of duplication.
HUMAN_PREFIX = "<s>[INST] "
HUMAN_SUFFIX = " [/INST]"
AI_PREFIX = " "
AI_SUFFIX = " </s>"


class Llama2PromptTemplate(ChatPromptTemplate):
    """Llama 2 prompt template.
    Llama 2 has a very special prompt format,
    in which the system message is 'squashed' into the first user message.
    See <https://github.com/facebookresearch/llama/blob/6c7fe276574e78057f917549435a2554000a876d/llama/generation.py#L213>
    and <https://huggingface.co/TheBloke/Llama-2-13B-chat-GPTQ/discussions/5>
    """

    def format_prompt(self, **kwargs: Any) -> PromptValue:
        """Format prompt."""
        messages = self.format_messages(**kwargs)
        return Llama2PromptValue(messages=messages)


class Llama2PromptValue(FlexPromptValue):
    system_prefix: Literal["<<SYS>>"] = "<<SYS>>"
    system_suffix: Literal["<</SYS>>"] = "<</SYS>>"
    human_prefix: Literal["<s>[INST] "] = "<s>[INST] "
    human_suffix: Literal[" [/INST]"] = " [/INST]"
    ai_prefix: Literal[" "] = " "
    ai_suffix: Literal[" </s>"] = " </s>"

    def to_string(self) -> str:
        """Return prompt as string."""
        if not isinstance(self.messages[0], SystemMessage):
            msgs = self.messages
        else:
            msgs = [
                BaseMessage(
                    type=self.messages[1].type,
                    content=f"{self.system_prefix}\n{self.messages[0].content}\n{self.system_suffix}\n\n{self.messages[1].content}",
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
        return msgs
