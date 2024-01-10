
import os
os.environ["OPENAI_API_KEY"] = "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"

import asyncio
from typing import AsyncIterable

from fastapi import FastAPI,Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage,AIMessage
from pydantic import BaseModel

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    content: str
    role: str
async def send_message(messages= list[Message]) -> AsyncIterable[str]:
    callback = AsyncIteratorCallbackHandler()
    model = ChatOpenAI(
        streaming=True,
        verbose=True,
        callbacks=[callback],
    )
    msgs = []
    for x in messages:
        i = Message(**x)
        content = i.content
        role = i.role
        if role in ["human","user"]:
            msg = HumanMessage(content=content)
        else:
            msg = AIMessage(content=content)
        msgs.append(msg)

    task = asyncio.create_task(
        model.agenerate(messages=[msgs])
    )
    try:
        async for token in callback.aiter():
            yield token
    except Exception as e:
        print(f"Caught exception: {e}")
    finally:
        callback.done.set()
    await task
@app.post("/sqlchat")
async def stream_chat(request:Request):
    msgs = await request.json()
    msgs = msgs["messages"]
    generator = send_message(messages=msgs)
    return StreamingResponse(generator, media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)
