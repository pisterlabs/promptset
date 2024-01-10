# Copyright (c) 2023 Rocket Science AG, Switzerland

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Tools for maintaining a chat session (i.e. LLM context with messages)."""

from __future__ import annotations

import asyncio
import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Protocol

import aioconsole  # type: ignore[import]
import openai
from loguru import logger
from overrides import override

from rrosti.llm_api import openai_api
from rrosti.utils import misc
from rrosti.utils.config import config

ERROR_SLEEP_SECONDS = 2


class Importance(enum.IntEnum):
    """Importance of a message."""

    NOISE = 0
    LOW = 10
    MEDIUM = 20
    HIGH = 30


@dataclass
class Message:
    """A message in a chat session."""

    role: Literal["user", "assistant", "system"]
    text: str

    # To manage the size of the context, we may need to trim messages.
    # 1. Each message has an importance score; when the context is full, we always trim the earliest
    #    message with the lowest importance score.
    # 2. Messages can have an optional ttl (time-to-live) field which gets decremented at each
    #    user input. When the ttl reaches 0, the message is removed.
    importance: Importance

    ttl: int | None = None

    cost: float | None = None
    time_used: float | None = None

    def __post_init__(self) -> None:
        assert not (self.role == "user" and self.cost is not None), "User messages cannot have a cost"

    def as_user_message(self) -> Message:
        """
        Return a user message with the same text as this message, with no cost.
        """
        return Message(role="user", text=self.text, importance=self.importance)

    def to_string(self, agent: str | None = None) -> str:
        meta: list[str] = [self.role]
        if self.cost is not None:
            meta.append(f"{self.cost:.5f} USD")
        if self.time_used is not None:
            meta.append(f"time={self.time_used:.2f}")
        if self.ttl is not None:
            meta.append(f"ttl={self.ttl}")
        meta.append(f"importance={self.importance}")
        meta_str = ", ".join(meta)
        header = f"{agent} ({meta_str})" if agent else meta_str
        lines = []
        lines.append("-" * 5 + " " + header + " " + "-" * (70 - 7 - len(header)))
        lines.append(self.text.strip())
        lines.append("-" * 70)
        return "\n".join(lines)


async def get_user_input(prompt: str = "", end_line: str | None = None) -> str:
    """Get user input from the console, optionally ending on a specific line."""
    lines: list[str] = []
    while True:
        line = await aioconsole.ainput(f"{prompt}> ")
        if end_line is not None and line.strip() == end_line:
            break
        lines.append(line)
        if end_line is None:
            break

    return "\n".join(lines)


class LLM(ABC):
    """A language model."""

    @abstractmethod
    async def chat_completion(self, messages: list[Message], agent_name: str = "", model: str | None = None) -> Message:
        ...


class OpenAI(LLM):
    """OpenAI chat language model."""

    temperature: float
    prompt_tokens: int
    completion_tokens: int
    cost: float
    openai_provider: openai_api.OpenAIApiProvider

    def __init__(self, openai_provider: openai_api.OpenAIApiProvider, temperature: float = 1.0) -> None:
        self.openai_provider = openai_provider
        self.temperature = temperature
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cost = 0.0

    @override
    async def chat_completion(
        self,
        messages: list[Message],
        agent_name: str = "",
        model: str | None = None,
        msg_importance: Importance = Importance.MEDIUM,
    ) -> Message:
        del agent_name  # intentionally unused parameter

        if model is None:
            model = config.openai_api.chat_completion_model

        while True:
            try:
                start_time = asyncio.get_event_loop().time()
                resp = await self.openai_provider.acreate_chat_completion(
                    messages=[dict(role=message.role, content=message.text) for message in messages],
                    max_tokens=config.openai_api.completion_max_tokens,
                    model=model,
                )
                end_time = asyncio.get_event_loop().time()
                elapsed_time = end_time - start_time
                break
            except (
                openai.error.RateLimitError,
                openai.error.Timeout,
                openai.error.APIConnectionError,
                openai.error.APIError,
                openai.error.ServiceUnavailableError,
            ) as e:
                logger.error("OpenAI API error: {}. Sleeping for {} seconds...", e, ERROR_SLEEP_SECONDS)
                await asyncio.sleep(ERROR_SLEEP_SECONDS)

        prompt_tokens = resp["usage"]["prompt_tokens"]
        completion_tokens = resp["usage"]["completion_tokens"]

        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens

        model_cost = config.openai_api.model_cost[
            model if model is not None else config.openai_api.chat_completion_model
        ]

        # TODO: handle prompts that did not return any content at all (currently it crashes the process and starts a new
        # session with the client without any Warning
        text: str | None = resp["choices"][0]["message"].get("content")
        if not text:
            # If nothing is returned, the query (or answer) got blocked by Azure. This is the identifier so that the
            # state machine can give the user an answer explaining it.
            text = "$$$error$$$"

        return Message(
            role="assistant",
            text=text,
            importance=msg_importance,
            cost=model_cost.calculate(prompt_tokens, completion_tokens),
            time_used=elapsed_time,
        )


class UserInputLLM(LLM):
    """LLM that asks the user for input. For testing."""

    @override
    async def chat_completion(self, messages: list[Message], agent_name: str = "", model: str | None = None) -> Message:
        for message in messages:
            print("-" * 5 + " " + message.role + " " + "-" * (40 - 7 - len(message.role)))
            print(message.text)
        print("-" * 40)
        print("Enter message. '.' on a line alone finishes.")
        text = await get_user_input(agent_name, end_line=".")
        return Message(role="user", importance=Importance.HIGH, text=text)


class MessageCallback(Protocol):
    def __call__(self, message: Message, agent: str | None, quiet: bool) -> None:
        ...


class ChatSession:
    """
    A chat session (i.e. LLM context, consisting of messages).

    Also contains functionality to prune messages to prevent the LLM context from becoming too long.
    """

    messages: list[Message]
    llm: LLM
    _name: str  # A human-readable name. Could be e.g. a name of an agent. May be empty string.

    _message_callback: MessageCallback | None

    def __init__(self, llm: LLM, name: str = "", callback: MessageCallback | None = None) -> None:
        self.messages = []
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.llm = llm
        self._name = name
        self._message_callback = callback

    def _prune_unimportant_message(self) -> None:
        """Prune the least important message."""
        assert self.messages, "No messages to prune"
        min_importance = min(message.importance for message in self.messages)
        for message in self.messages:
            if message.importance == min_importance:
                logger.info(
                    "Pruning unimportant message (importance={}): {}",
                    message.importance,
                    misc.truncate_string(message.text),
                )
                self.messages.remove(message)
                return

    def decrease_ttls(self) -> None:
        """Decrease ttls and kill messages with expired ttls."""

        new_messages = []

        for message in self.messages:
            if message.ttl is not None:
                message.ttl -= 1
                logger.info("Decreasing ttl of message to {}: {}", message.ttl, misc.truncate_string(message.text))

            if message.ttl is not None and message.ttl <= 0:
                logger.info("Killing message with expired ttl: {}", misc.truncate_string(message.text))
            else:
                new_messages.append(message)

        self.messages = new_messages

    def add_message(
        self,
        message: Message,
        quiet: bool = False,
    ) -> None:
        """Add a message to the chat session."""

        assert message.ttl is None or message.ttl > 0, "llm_ttl must be None or positive"

        self.messages.append(message)
        logger.info(
            "Message({}): {}",
            message.role,
            misc.truncate_string(message.text),
        )

        if self._message_callback is not None:
            self._message_callback(message, self._name, quiet)

    async def generate_response(self, model: str | None = None) -> Message:
        while True:
            try:
                resp = await self.llm.chat_completion(self.messages, agent_name=self._name, model=model)
                break
            except openai.error.InvalidRequestError as e:
                if "maximum context length" in str(e):
                    logger.info("Context length exceeded: {}", str(e))
                    self._prune_unimportant_message()
                    continue
                raise

        self.add_message(resp)
        return resp
