import asyncio
from typing import Any
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from langchain.schema import LLMResult


class AsyncCallbackHandler_LLM(AsyncIteratorCallbackHandler):
    """
    A custom callback handler for asynchronous streaming of tokens from a large language model (LLM).

    Attributes:
        content (str): Accumulates the tokens received from the LLM.
        final_answer (bool): Flag indicating if the final answer has been reached (not used in this implementation).

    Args:
        delay (float): Delay in seconds before processing each new token. Defaults to 1.0.

    This class extends AsyncIteratorCallbackHandler and is designed to handle new tokens and the end of an LLM's response in a streaming context.
    """

    content: str = ""
    final_answer: bool = False

    def __init__(self, delay: float = 1.0) -> None:
        super().__init__()
        self.delay = delay

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        await asyncio.sleep(self.delay)
        self.content += token
        self.queue.put_nowait(token)

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        self.done.set()


async def llm_run_call_no_stream(llm: object, query: str, context: str):
    """
    Executes a non-streaming call to the large language model (LLM).

    Args:
        llm (object): The LLM object.
        query (str): The input query to be processed by the LLM.
        context (str): The context in which the query should be interpreted.

    Returns:
        The result from processing the input query by the LLM in the given context.

    This function makes an asynchronous call to the LLM with the given query and context.
    """
    return await llm.acall(inputs={"input": query, "context": context})


async def llm_run_acall(
    llm: object, query: str, context: str, stream_it: AsyncCallbackHandler_LLM
):
    """
    Runs an asynchronous call with streaming enabled using a custom callback handler for a large language model (LLM).

    Args:
        llm (object): The LLM object.
        query (str): The input query to be processed by the LLM.
        context (str): The context in which the query should be interpreted.
        stream_it (AsyncCallbackHandler_LLM): The callback handler for streaming the response.

    This function sets the callback handler for the LLM and initiates an asynchronous call with the given query and context.
    """
    # assign callback handler
    llm.llm.callbacks = [stream_it]
    # now query
    await llm.acall(inputs={"input": query, "context": context})


async def llm_create_gen(
    llm: object, query: str, context: str, stream_it: AsyncCallbackHandler_LLM
):
    """
    Creates an asynchronous generator for streaming tokens from a large language model (LLM).

    Args:
        llm (object): The LLM object.
        query (str): The input query to be processed by the LLM.
        context (str): The context in which the query should be interpreted.
        stream_it (AsyncCallbackHandler_LLM): The callback handler for streaming the response.

    Returns:
        An asynchronous generator yielding tokens from the LLM.

    This function initiates an asynchronous call with streaming and yields tokens as they are received.
    """
    task = asyncio.create_task(llm_run_acall(llm, query, context, stream_it))

    async for token in stream_it.aiter():
        yield token
    await task
