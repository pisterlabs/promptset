from typing import Any
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.schema.output import LLMResult
from langchain.agents.openai_functions_multi_agent.base import OpenAIMultiFunctionsAgent
from langchain.callbacks.base import AsyncCallbackHandler
import asyncio


class AsyncOpenAIFunctionAgentCallbackHandler(AsyncIteratorCallbackHandler):
    """Callback handler that returns an async iterator."""

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        ai_message_content = response.generations[0][0].message.content

        if ai_message_content != "":
            self.done.set()

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.queue.put_nowait(token)
