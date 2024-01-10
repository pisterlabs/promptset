from typing import Dict
from typing import List
from typing import Optional
from typing import Any

from langchain.memory.token_buffer import BaseLanguageModel  # bypass refactor
from langchain.memory.chat_memory import BaseChatMemory
from langchain.schema import get_buffer_string


class EmbeddingConversationMemory(BaseChatMemory):
    """Buffer for storing conversation memory. This buffer will store k messages
    from the start of the conversation and k messages before the end of the conversation
    Set k to None to store the full conversation.

    Args:
        human_prefix: Prefix to use for human messages.
        ai_prefix: Prefix to use for ai messages.
        memory_key: Key to use for memory.
        llm: Language model to use.
        max_token_limit: Maximum number of tokens to store in memory. If empty, tokens are not limited.
        k: Number of messages to store from the start and end of the conversation.
    """

    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    memory_key: str = "history"
    llm: BaseLanguageModel
    max_token_limit: Optional[int] = None
    k: int = 5

    @property
    def buffer(self):
        """String buffer of memory."""
        return self.chat_memory.messages

    @property
    def memory_variables(self):
        """Will always return list of memory variables."""
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]):
        """Return history buffer."""
        buffer: Any = self.buffer
        if self.k is not None and len(self.buffer) >= self.k * 2:
            buffer_inds = list(range(len(buffer)))
            buffer_inds = buffer_inds[0 : self.k] + buffer_inds[-self.k :]
            buffer_inds = list(sorted(set(buffer_inds)))
            buffer = [buffer[i] for i in buffer_inds]
        if not self.return_messages:
            buffer = get_buffer_string(
                buffer,
                human_prefix=self.human_prefix,
                ai_prefix=self.ai_prefix,
            )
        return {self.memory_key: buffer}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]):
        """Save context from this conversation to buffer. Pruned."""
        super().save_context(inputs, outputs)
        if self.max_token_limit is None:
            return
        # Prune buffer if it exceeds max token limit
        buffer = self.chat_memory.messages
        curr_buffer_length = self.llm.get_num_tokens_from_messages(buffer)
        if curr_buffer_length > self.max_token_limit:
            pruned_memory = []
            while curr_buffer_length > self.max_token_limit:
                # we pop the ends
                pruned_memory.append(buffer.pop(-1))
                curr_buffer_length = self.llm.get_num_tokens_from_messages(buffer)
