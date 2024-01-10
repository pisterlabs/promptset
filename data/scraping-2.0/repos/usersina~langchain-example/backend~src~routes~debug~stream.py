import asyncio
import json
import os
import random
from typing import Any, AsyncGenerator, TypedDict

from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from langchain.llms import OpenAI
from quart import Blueprint, request

from utils.generate import generate_random_string

page = Blueprint(
    f"{os.path.dirname(__file__).replace(os.path.sep, '-')}-{os.path.splitext(os.path.basename(__file__))[0]}",
    __name__,
)


# Define the expected input type
class Input(TypedDict):
    prompt: str


@page.route("/", methods=["POST"])
async def stream_text() -> tuple[AsyncGenerator[str, Any], dict[str, str]]:
    data: Input = await request.get_json()

    prompt = data["prompt"]

    async def ask_question_async():
        handler = AsyncIteratorCallbackHandler()
        llm = OpenAI(streaming=True, callbacks=[handler])
        asyncio.create_task(llm.agenerate([prompt]))
        async for i in handler.aiter():
            yield i

    return ask_question_async(), {"Content-Type": "text/event-stream"}


@page.route("/dummy", methods=["GET"])
async def stream_dummy():
    chunks_amount_arg = request.args.get("chunks_amount")
    chunks_amount = int(chunks_amount_arg) if chunks_amount_arg else 5000

    def generate_output():
        for i in range(chunks_amount):
            print(i)
            yield generate_random_string(100)
            yield "\n===============================================\n"

    return generate_output(), {"Content-Type": "text/event-stream"}


@page.route("/dummy/json", methods=["GET"])
async def stream_dummy_json():
    chunks_amount_arg = request.args.get("chunks_amount")
    chunks_amount = int(chunks_amount_arg) if chunks_amount_arg else 5000

    chunk_max_size = 32  # full chunk size ('{"word":"some-word"}')
    chunk_content_size = 21  # generated content size ('some-word')
    chunk_wrapper_size = 11  # wrapper size ('{"word":""}')

    def generate_output():
        for i in range(chunks_amount):
            print(f"Iteration: {i}")
            # Generate a random word between 5 and 80 utf-8 characters (bytes)
            random_word = generate_random_string(random.randint(5, 80))
            if (len(random_word) + chunk_wrapper_size) > chunk_max_size:
                # If the chunk and wrapper size exceeds the max size, further chunk it
                for i in range(0, len(random_word), chunk_content_size):
                    sub_chunk = random_word[i : i + chunk_content_size]
                    response = json.dumps({"word": sub_chunk}, separators=(",", ":"))
                    print(f"Sub Chunk Size: {len(response)}")
                    yield response
                    yield "\n"
            else:
                # No need to chunk, send as is
                response = json.dumps({"word": random_word}, separators=(",", ":"))
                print(f"Chunk Size: {len(response)}")
                yield response
                yield "\n"

            yield "===============================================\n"
            print("\n")

    return generate_output(), {"Content-Type": "text/event-stream"}


# POC on the importance of streaming
@page.route("/dummy/none", methods=["GET"])
async def streamTextNone() -> str:
    def generate_output():
        a = ""
        for i in range(10000):
            print(i)
            a = (
                a
                + generate_random_string(100)
                + "\n===============================================\n"
            )
        return a

    output = generate_output()
    return output
