"""
Chatterbox is a high-level interface for instanced conversations with ChatGPT.
"""
import asyncio
from functools import cached_property

import cachetools
import tiktoken

from .client import OpenAIClient
from .models import ChatMessage


class Chatterbox:
    def __init__(
        self,
        client: OpenAIClient,
        system_prompt: str,
        always_include_messages: list[ChatMessage] = None,
        model="gpt-3.5-turbo",
        desired_response_tokens: int = 450,  # roughly the size of a discord message
        max_context_size: int = 4096,  # depends on model,
        chat_history: list[ChatMessage] = None,
        **hyperparams
    ):
        hyperparams["max_tokens"] = desired_response_tokens
        self.client = client
        self.tokenizer = None
        self.system_prompt = system_prompt.strip()
        self.model = model
        self.desired_response_tokens = desired_response_tokens
        self.max_context_size = max_context_size
        self.always_include_messages = [ChatMessage.system(self.system_prompt)] + (always_include_messages or [])
        self.chat_history: list[ChatMessage] = chat_history or []
        self.hyperparams = hyperparams

        # async to prevent generating multiple responses missing context
        self.lock = asyncio.Lock()

        # cache
        self._oldest_idx = 0
        self._message_tokens = cachetools.FIFOCache(256)

    def _load_tokenizer(self):
        """
        Load the tokenizer (from the internet if first run). See
        https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
        """
        try:
            self.tokenizer = tiktoken.encoding_for_model(self.model)
        except KeyError:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    @cached_property
    def system_prompt_len(self):
        return len(self.tokenizer.encode(self.system_prompt)) + 5

    def message_token_len(self, message: ChatMessage):
        """Returns the number of tokens used by a list of messages."""
        try:
            return self._message_tokens[message.content]
        except KeyError:
            mlen = len(self.tokenizer.encode(message.content)) + 5  # ChatML = 4, role = 1
            self._message_tokens[message.content] = mlen
            return mlen

    async def get_truncated_chat_history(self):
        """
        Returns a list of messages such that the total token count in the messages is less than
        (4096 - desired_response_tokens).
        Always includes the system prompt plus any always_include_messages.
        """
        reversed_history = []
        always_len = sum(self.message_token_len(m) for m in self.always_include_messages)
        remaining = self.max_context_size - (always_len + self.desired_response_tokens)
        for idx in range(len(self.chat_history) - 1, self._oldest_idx - 1, -1):
            message = self.chat_history[idx]
            message_len = self.message_token_len(message)
            remaining -= message_len
            if remaining > 0:
                reversed_history.append(message)
            else:
                self._oldest_idx = idx + 1
                break
        return self.always_include_messages + reversed_history[::-1]

    # === main entrypoints ===
    async def load_tokenizer(self):
        await asyncio.get_event_loop().run_in_executor(None, self._load_tokenizer)

    async def chat_round(self, query: str, **kwargs) -> str:
        async with self.lock:
            # get the user's chat input
            self.chat_history.append(ChatMessage.user(query.strip()))

            # get the context
            messages = await self.get_truncated_chat_history()

            # get the model's output, save it to chat history
            completion = await self.client.create_chat_completion(
                model=self.model, messages=messages, **self.hyperparams, **kwargs
            )
            self._message_tokens[completion.text] = completion.usage.completion_tokens + 5
            self.chat_history.append(completion.message)
            return completion.text
