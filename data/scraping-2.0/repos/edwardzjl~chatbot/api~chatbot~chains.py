from typing import Any

from langchain.chains import LLMChain
from langchain.memory.chat_memory import BaseChatMemory

from chatbot.memory import ChatbotMemory


class LLMConvChain(LLMChain):
    """Conversation chain that persists message separately on chain start and end."""

    user_input_variable: str

    def prep_inputs(self, inputs: dict[str, Any] | Any) -> dict[str, str]:
        """Override this method to persist input on chain starts.
        We need to separatly save the input and output on chain starts and ends.
        """
        inputs = super().prep_inputs(inputs)
        # we need to access the history so we need to ensure it's BaseChatMemory and then we can access it by memory.chat_memory
        if self.memory is not None and isinstance(self.memory, ChatbotMemory):
            message = inputs[self.user_input_variable]
            self.memory.history.add_user_message(message)
        if self.memory is not None and isinstance(self.memory, BaseChatMemory):
            message = inputs[self.user_input_variable]
            self.memory.chat_memory.add_user_message(message)
        return inputs

    def prep_outputs(
        self,
        inputs: dict[str, str],
        outputs: dict[str, str],
        return_only_outputs: bool = False,
    ) -> dict[str, str]:
        """Override this method to disable saving context to memory.
        We need to separatly save the input and output on chain starts and ends.
        """
        self._validate_outputs(outputs)
        # we need to access the history so we need to ensure it's BaseChatMemory and then we can access it by memory.chat_memory
        if self.memory is not None and isinstance(self.memory, ChatbotMemory):
            text = outputs[self.output_key]
            self.memory.history.add_ai_message(text)
        if self.memory is not None and isinstance(self.memory, BaseChatMemory):
            text = outputs[self.output_key]
            self.memory.chat_memory.add_ai_message(text)
        if return_only_outputs:
            return outputs
        else:
            return {**inputs, **outputs}
