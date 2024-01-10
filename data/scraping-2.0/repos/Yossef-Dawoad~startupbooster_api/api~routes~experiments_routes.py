import asyncio
import logging
from collections.abc import AsyncGenerator, Generator

from fastapi import APIRouter, Request
from langchain import HuggingFaceTextGenInference
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from sse_starlette import EventSourceResponse

log = logging.getLogger("api-routes-logs")

router = APIRouter(
    prefix="/api/experiments",
    tags=["Experiments Apis"],
)

# Constants
REPO_ID = "tiiuae/falcon-7b-instruct"
INFER_SERVER_URL = f"https://api-inference.huggingface.co/models/{REPO_ID}"

# ////////


async def generate_streamed_response(
    message: str,
    req_obj: Request,
) -> AsyncGenerator[str, None]:
    """Generate a response for a given message using HuggingFaceTextGenInference.

    Args:
        message: The input message to generate a response for.
        request: The request object to check if the client is disconnected.

    Yields:
        A token of the generated response as a string.
    """
    callback_handler = AsyncIteratorCallbackHandler()
    llm = HuggingFaceTextGenInference(
        # inference_server_url="http://localhost:8010/",
        inference_server_url=INFER_SERVER_URL,
        max_new_tokens=512,
        top_k=10,
        top_p=0.95,
        typical_p=0.95,
        temperature=0.9,
        repetition_penalty=1.03,
        stream=True,
        callbacks=[callback_handler],
    )

    run = asyncio.create_task(llm.agenerate(prompts=[message]))
    async for token in callback_handler.aiter():
        # if the user dis-miss the request strop the generation.
        if await req_obj.is_disconnected():
            break
        yield token
    await run


@router.get("/stream-response")
async def message_stream(request: Request, user_message: str) -> EventSourceResponse:
    """Send a stream of generated response tokens for a given user message.

    Args:
        request: The request object to pass to the generator function.
        user_message: The user message to generate a response for.

    Returns:
        An EventSourceResponse object that streams the response tokens.
    """
    return EventSourceResponse(
        # sending request object to check if the client disconncted stop
        generate_streamed_response(user_message, req_obj=request),
        media_type="text/event-stream",
    )


@router.get("/jokes")
def getjokes(request: Request) -> EventSourceResponse:
    # the streaming function
    def get_massages() -> Generator[str]:
        yield "this is repeating message"

    async def sse_event() -> AsyncGenerator:
        while True:
            # if the client disconnect close
            if await request.is_disconnected():
                log.warn("user disconnected, terminating sse_event")
                break
            for message in get_massages():
                yield {"data": message}
            await asyncio.sleep(1)

    # retrun the server-side event
    return EventSourceResponse(sse_event())
