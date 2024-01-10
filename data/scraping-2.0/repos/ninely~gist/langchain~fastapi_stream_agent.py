import asyncio
import os
from typing import AsyncIterable, Awaitable

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain.agents import AgentType
from langchain.agents import initialize_agent
from langchain.agents import load_tools
from langchain.llms import OpenAI
from langchain.utilities import DuckDuckGoSearchAPIWrapper

from callback import streaming_aiter_final_only
from tools import proxy_ddg_search

# Two ways to load env variables
# 1.load env variables from .env file
load_dotenv()

# 2.manually set env variables
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = ""

app = FastAPI()


async def do_agent() -> AsyncIterable[str]:
    question = "Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?"

    callback = streaming_aiter_final_only.AsyncFinalIteratorCallbackHandler(stream_prefix=True)

    llm = OpenAI(
        temperature=0,
    )

    stream_llm = OpenAI(
        temperature=0,
        streaming=True,
        verbose=True,
        callbacks=[callback],
    )
    tools = load_tools(["llm-math"], llm=stream_llm)
    tools.append(proxy_ddg_search.ProxyDuckDuckGoSearchRun(api_wrapper=DuckDuckGoSearchAPIWrapper()))
    agent = initialize_agent(tools, stream_llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    async def wrap_done(fn: Awaitable, event: asyncio.Event):
        """Wrap an awaitable with an event to signal when it's done or an exception is raised."""
        try:
            await fn
        except Exception as e:
            # TODO: handle exception
            print(f"Caught exception: {e}")
        finally:
            # Signal the aiter to stop.
            event.set()

    # Begin a task that runs in the background.
    task = asyncio.create_task(wrap_done(
        agent.arun(question),
        callback.done),
    )

    async for token in callback.aiter():
        # Use server-sent-events to stream the response
        yield f"data: {token}\n\n"

    await task


@app.post("/stream/agent")
def stream():
    return StreamingResponse(do_agent(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(host="0.0.0.0", port=8000, app=app)
