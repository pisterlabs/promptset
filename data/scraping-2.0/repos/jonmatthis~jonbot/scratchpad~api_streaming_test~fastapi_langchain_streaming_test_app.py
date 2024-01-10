"""This is an example of how to use async langchain with fastapi and return a streaming response.
The latest version of Langchain has improved its compatibility with asynchronous FastAPI,
making it easier to implement streaming functionality in your applications.
"""
import asyncio
import os
from typing import AsyncIterable, Awaitable

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain import LLMChain, PromptTemplate
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableSequence
from pydantic import BaseModel

# Two ways to load env variables
# 1.load env variables from .env file
load_dotenv()

# 2.manually set env variables
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = ""

app = FastAPI()


def make_chain_with_expression_language() -> RunnableSequence:
    prompt = PromptTemplate.from_template("tell me a joke about {topic}")
    model = ChatOpenAI()
    chain = model | prompt
    return chain


async def send_message_expression_chain(message: str) -> AsyncIterable[str]:
    # chain = make_chain_with_expression_language()
    model = ChatOpenAI()
    prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")
    chain = prompt | model
    async for token in chain.astream({"topic": message}):
        # Use server-sent-events to stream the response
        print(f"Sending token: {token.content}")
        yield f"data: {token.content}\n\n"


def make_chain(callback: AsyncIteratorCallbackHandler) -> LLMChain:
    prompt = PromptTemplate(
        template="tell me a joke about {topic}",
        input_variables=["topic"],
    )

    llm = ChatOpenAI(
        streaming=True,
        verbose=True,
        callbacks=[callback],
    )
    return LLMChain(llm=llm, prompt=prompt)


async def send_message_trad_chain(message: str) -> AsyncIterable[str]:
    callback = AsyncIteratorCallbackHandler()
    chain = make_chain(callback)

    async def wrap_done(fn: Awaitable, event: asyncio.Event):
        """Wrap an awaitable with a event to signal when it's done or an exception is raised."""
        try:
            await fn
        except Exception as e:
            # TODO: handle exception
            print(f"Caught exception: {e}")
        finally:
            # Signal the aiter to stop.
            event.set()

    # Begin a task that runs in the background.
    task = asyncio.create_task(
        wrap_done(chain.agenerate(input_list=[{"topic": message}]), callback.done),
    )

    async for token in callback.aiter():
        # Use server-sent-events to stream the response
        print(f"Sending token: {token}")
        yield f"data: {token}\n\n"

    await task


class StreamRequest(BaseModel):
    """Request body for streaming."""

    message: str


@app.post("/stream_trad")
def stream(body: StreamRequest):
    return StreamingResponse(
        send_message_trad_chain(body.message), media_type="text/event-stream"
    )


@app.post("/stream_expression")
def stream(body: StreamRequest):
    return StreamingResponse(
        send_message_expression_chain(body.message), media_type="text/event-stream"
    )


if __name__ == "__main__":
    uvicorn.run(host="localhost", port=8000, app=app)
