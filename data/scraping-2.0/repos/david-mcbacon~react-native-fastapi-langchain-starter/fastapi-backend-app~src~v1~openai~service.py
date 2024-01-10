import asyncio
import os
from typing import AsyncIterable, Awaitable
from dotenv import load_dotenv
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from typing import List
import json
import logging

from src.v1.openai.schemas import MessageOpenAi, StreamRequestOpenAi

load_dotenv()


def transform_messages_to_langchain(messages: List[MessageOpenAi]) -> list:
    """Transform messages to langchain format."""
    formatted_messages = []
    for message in messages:
        if message.role == "user":
            formatted_messages.append(HumanMessage(content=message.content))
        elif message.role == "assistant":
            formatted_messages.append(AIMessage(content=message.content))

    return formatted_messages


async def send_message(body: StreamRequestOpenAi) -> AsyncIterable[str]:
    callback = AsyncIteratorCallbackHandler()
    model = ChatOpenAI(
        streaming=True,
        verbose=True,
        callbacks=[callback],
        api_key=os.getenv("OPENAI_API_KEY"),
        model=body.model,
    )

    async def wrap_done(fn: Awaitable, event: asyncio.Event, max_retries: int = 3):
        """
        This function wraps an awaitable function with an event to signal when it's done or an exception is raised.

        Args:
            fn (Awaitable): The awaitable function to be wrapped.
            event (asyncio.Event): An event to signal when the function is done.
            max_retries (int, optional): The maximum number of retries in case of a TimeoutError. Defaults to 3.

        The function does the following:
        - Tries to await the function and handles two types of exceptions: asyncio.TimeoutError and general exceptions.
        - In case of asyncio.TimeoutError, it retries the function up to max_retries times.
        - In case of any other exception, it logs the exception and re-raises it.
        - After the function is done or an exception is raised, it sets the event to signal that the awaitable function is done.
        """
        retries = 0
        while retries < max_retries:
            try:
                await fn
                break

            except asyncio.TimeoutError as e:
                retries += 1
                logging.error(f"Timeout Error: {e}. Retry {retries} of {max_retries}")

            except Exception as e:
                logging.error(f"Caught exception: {e}")
                raise e

            finally:
                # Signal the aiter to stop.
                event.set()

    langchain_messages = transform_messages_to_langchain(body.messages)

    # Begin a task to run in the background
    task = asyncio.create_task(
        wrap_done(
            model.agenerate(messages=[langchain_messages]),
            callback.done,
        ),
    )

    async for token in callback.aiter():
        # Use server-sent-events to stream the response from the model
        yield f"data: {json.dumps({'content': token})}\n\n"

    await task

    # Send the final message to indicate that the streaming is over
    yield "data: [DONE]\n\n"
