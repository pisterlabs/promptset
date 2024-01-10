from abc import ABC
from typing import Any, AsyncIterator, List, TypedDict

import openai
from aidial_sdk.chat_completion import Role
from aiohttp import ClientSession
from pydantic import BaseModel


class ReasonLengthException(Exception):
    pass


class Message(BaseModel):
    role: Role
    content: str

    def to_openai_message(self) -> dict[str, str]:
        return {"role": self.role.value, "content": self.content}

    @classmethod
    def system(cls, content):
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def user(cls, content):
        return cls(role=Role.USER, content=content)

    @classmethod
    def assistant(cls, content):
        return cls(role=Role.ASSISTANT, content=content)


class Usage(TypedDict):
    prompt_tokens: int
    completion_tokens: int


class ExtraResultsCallback:
    def on_discarded_messages(self, discarded_messages: int):
        pass

    def on_prompt_tokens(self, prompt_tokens: int):
        pass


async def _flush_stream(stream: AsyncIterator[str]):
    try:
        async for _ in stream:
            pass
    except ReasonLengthException:
        pass


class ModelClient(ABC):
    def __init__(
        self,
        model_args: dict[str, Any],
        buffer_size: int,
    ):
        self.model_args = model_args
        self.buffer_size = buffer_size

        self._total_prompt_tokens: int = 0
        self._total_completion_tokens: int = 0

    async def agenerate(
        self,
        messages: List[Message],
        extra_results_callback: ExtraResultsCallback | None = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        async with ClientSession(read_bufsize=self.buffer_size) as session:
            openai.aiosession.set(session)

            model_result = await openai.ChatCompletion.acreate(
                messages=[message.to_openai_message() for message in messages],
                **self.model_args | kwargs,
            )

            finish_reason_length = False
            async for chunk in model_result:  # type: ignore
                usage: Usage | None = chunk.get("usage")
                if usage:
                    prompt_tokens = usage["prompt_tokens"]
                    self._total_prompt_tokens += prompt_tokens
                    self._total_completion_tokens += usage["completion_tokens"]
                    if extra_results_callback:
                        extra_results_callback.on_prompt_tokens(prompt_tokens)

                if extra_results_callback:
                    discarded_messages: int | None = chunk.get(
                        "statistics", {}
                    ).get("discarded_messages")
                    if discarded_messages is not None:
                        extra_results_callback.on_discarded_messages(
                            discarded_messages
                        )

                choice = chunk["choices"][0]
                text = choice["delta"].get("content")
                if text:
                    yield text

                if choice.get("finish_reason") == "length":
                    finish_reason_length = True

            if finish_reason_length:
                raise ReasonLengthException()

    # TODO: Use a dedicated endpoint for counting tokens.
    #  This request may throw an error if the number of tokens is too large.
    async def count_tokens(self, messages: list[Message]) -> int:
        class PromptTokensCallback(ExtraResultsCallback):
            def __init__(self):
                self.token_count: int | None = None

            def on_prompt_tokens(self, prompt_tokens: int):
                self.token_count = prompt_tokens

        callback = PromptTokensCallback()
        await _flush_stream(
            self.agenerate(
                messages, extra_results_callback=callback, max_tokens=1
            )
        )
        if callback.token_count is None:
            raise Exception("No token count received.")

        return callback.token_count

    # TODO: Use a dedicated endpoint for discarded_messages.
    async def get_discarded_messages(
        self, messages: list[Message], max_prompt_tokens: int
    ) -> int:
        class DiscardedMessagesCallback(ExtraResultsCallback):
            def __init__(self):
                self.message_count: int | None = None

            def on_discarded_messages(self, discarded_messages: int):
                self.message_count = discarded_messages

        callback = DiscardedMessagesCallback()
        await _flush_stream(
            self.agenerate(
                messages,
                extra_results_callback=callback,
                max_prompt_tokens=max_prompt_tokens,
                max_tokens=1,
            )
        )
        if callback.message_count is None:
            raise Exception("No message count received.")

        return callback.message_count

    @property
    def total_prompt_tokens(self) -> int:
        return self._total_prompt_tokens

    @property
    def total_completion_tokens(self) -> int:
        return self._total_completion_tokens
