from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Union, cast

from anthropic import AuthenticationError as AnthropicAuthenticationError
from langchain.callbacks.base import AsyncCallbackHandler
from langchain.schema.agent import AgentFinish
from langchain.schema.messages import BaseMessage
from langchain.schema.output import LLMResult
from openai.error import AuthenticationError as OpenAIAuthenticationError

from reflex_gptp.utils import OutputType

# TODO If used by two LLM runs in parallel this won't work as expected


class CustomAsyncIteratorCallbackHandler(AsyncCallbackHandler):
    """Callback handler that returns an async iterator."""

    queue: asyncio.Queue[tuple[OutputType, str, str | None]]

    @property
    def always_verbose(self) -> bool:
        return True

    def __init__(self) -> None:
        self.queue = asyncio.Queue()

    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        **kwargs: Any,
    ) -> None:
        pass

    async def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any) -> None:
        # If two calls are made in a row, this resets the state
        pass

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        if token is not None and token != "":
            self.queue.put_nowait((OutputType.TOKEN, token, None))

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        pass

    async def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        self.queue.put_nowait((OutputType.AGENT_FINISH, finish.return_values["output"], finish.log))

    async def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        message = str(error)
        if isinstance(error, AnthropicAuthenticationError | OpenAIAuthenticationError):
            message = "Authentication error. Please check your API key and try again."
        self.queue.put_nowait((OutputType.LLM_ERROR, message, None))

    async def on_tool_start(self, serialized: dict[str, Any], input_str: str, **kwargs) -> None:
        self.queue.put_nowait((OutputType.TOOL_START, serialized["name"], input_str))

    async def on_tool_end(self, output: str, name: str, **kwargs) -> None:
        self.queue.put_nowait((OutputType.TOOL_END, name, output))

    # TODO implement the other methods

    async def aiter(self) -> AsyncIterator[tuple[OutputType, str, str | None]]:  # noqa: A003
        while True:
            # Wait for the next token in the queue,
            # but stop waiting if the done event is set
            done, other = await asyncio.wait(
                [
                    # NOTE: If you add other tasks here, update the code below,
                    # which assumes each set has exactly one task each
                    asyncio.ensure_future(self.queue.get()),
                ],
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Cancel the other task
            if other:
                other.pop().cancel()

            # Extract the value of the first completed task
            res = cast(tuple[OutputType, str, str | None], done.pop().result())
            output_type = res[0]

            yield res

            # If the agent finished or there is an error, stop the loop
            if output_type in [
                OutputType.AGENT_FINISH,
                OutputType.LLM_ERROR,
                OutputType.INTERRUPT,
            ]:
                break
