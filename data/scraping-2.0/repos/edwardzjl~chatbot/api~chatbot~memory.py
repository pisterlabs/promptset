from typing import Any, Optional

from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory
from langchain.memory.utils import get_prompt_input_key
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.memory import BaseMemory
from langchain_core.messages import BaseMessage
from pydantic.v1 import Field, validator

from chatbot.history import ChatbotMessageHistory


class ChatbotMemory(BaseMemory):
    history: BaseChatMessageHistory = Field(default_factory=ChatMessageHistory)
    output_key: Optional[str] = None
    input_key: Optional[str] = None
    return_messages: bool = True
    memory_key: str = "history"  #: :meta private:
    k: int = 5
    """Number of messages to store in buffer."""

    @validator("k")
    def k_must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("k must be greater than 0")
        return v

    @property
    def buffer(self) -> str | list[BaseMessage]:
        """String buffer of memory."""
        return self.buffer_as_messages if self.return_messages else self.buffer_as_str

    @property
    def buffer_as_messages(self) -> list[BaseMessage]:
        """Exposes the buffer as a list of messages in case return_messages is False."""
        if isinstance(self.history, ChatbotMessageHistory):
            return self.history.windowed_messages(self.k)
        return self.history.messages[-self.k * 2 :] if self.k > 0 else []

    @property
    def buffer_as_str(self) -> str:
        # not going to support this
        raise NotImplementedError

    @property
    def memory_variables(self) -> list[str]:
        """Will always return list of memory variables.

        :meta private:
        """
        return [self.memory_key]

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Return history buffer."""
        return {self.memory_key: self.buffer}

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, str]) -> None:
        """Save context from this conversation to buffer."""
        input_str, output_str = self._get_input_output(inputs, outputs)
        self.history.add_user_message(input_str)
        self.history.add_ai_message(output_str)

    def clear(self) -> None:
        """Clear memory contents."""
        self.history.clear()

    def _get_input_output(
        self, inputs: dict[str, Any], outputs: dict[str, str]
    ) -> tuple[str, str]:
        if self.input_key is None:
            prompt_input_key = get_prompt_input_key(inputs, self.memory_variables)
        else:
            prompt_input_key = self.input_key
        if self.output_key is None:
            if len(outputs) != 1:
                raise ValueError(f"One output key expected, got {outputs.keys()}")
            output_key = list(outputs.keys())[0]
        else:
            output_key = self.output_key
        return inputs[prompt_input_key], outputs[output_key]
