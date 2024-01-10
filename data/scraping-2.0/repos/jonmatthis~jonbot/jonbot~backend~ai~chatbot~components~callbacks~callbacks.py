import asyncio
import logging
from typing import Union

from langchain.callbacks.base import AsyncCallbackHandler

from jonbot.system.setup_logging.get_logger import get_jonbot_logger

logger = get_jonbot_logger()


class AsyncQueueStuffingCallbackHandler(AsyncCallbackHandler):
    queue: asyncio.Queue = asyncio.Queue()
    stop_signal: str = "LLM_ENDED_STOPPING_NOW"

    async def on_llm_new_token(self, token: str, *args, **kwargs) -> None:
        """Run when a new token is generated."""
        logger.trace(
            f"Stuffing new token into queue: {token} - Queue size: {self.queue.qsize()}"
        )
        await self.queue.put(token)

    async def get_next_token(self) -> Union[str, None]:
        logging.trace(f"Getting next token from queue (size: {self.queue.qsize()})")
        if not self.queue.empty():
            token = await self.queue.get()
            logger.trace(
                f"Returning next token (`{token}`) from queue (size: {self.queue.qsize()})"
            )
            return token
        else:
            logger.trace(f"Queue is empty, returning None")
            return None

    async def on_llm_end(self, *args, **kwargs) -> None:
        """Run when chain ends running."""
        logger.trace(
            f"LLM ended, stuffing stop signal (self.stop_signal: {self.stop_signal}) into queue (size: {self.queue.qsize()})"
        )
        await self.queue.put(self.stop_signal)


class StreamingAsyncCallbackHandler(AsyncCallbackHandler):
    async def on_llm_new_token(self, token: str, *args, **kwargs) -> None:
        """Run when a new token is generated."""
        print("Hi! I just woke up. Your llm is generating a new token: '{token}'")
        yield f"lookit this token: {token} |"
