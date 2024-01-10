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
from fastapi.middleware.cors import CORSMiddleware

from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from pydantic import BaseModel

__version__ = "0.0.1"

load_dotenv()

app = FastAPI()

# CORS 설정 추가
app.add_middleware(
    CORSMiddleware,
    # allow_origins=["http://gears001.iptime.org:22280"],  # 특정 출처 허용
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메소드 허용
    allow_headers=["*"],  # 모든 헤더 허용
)

model = ChatOpenAI(
        streaming=True,
        verbose=True,
        # callbacks=[callback],
)

#%%

# /
@app.get("/")
def read_root():
    return {"version": __version__, "message": "it it lite chatbot server"}

# /qa
class QARequest(BaseModel):
    """Request body for QA."""
    question: str
@app.post("/qa")
def qa(_req: QARequest):
    return model.invoke(_req.question)

# /stream
class StreamRequest(BaseModel):
    """Request body for streaming."""
    message: str
@app.post("/stream")
def stream(_req: StreamRequest):
    
    async def send_message(message: str) -> AsyncIterable[str]:
        callback = AsyncIteratorCallbackHandler()
        
        model.callbacks = [callback]

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
        task = asyncio.create_task(wrap_done(
            model.agenerate(messages=[[HumanMessage(content=message)]]),
            callback.done),
        )
        
        async for token in callback.aiter():
            # Use server-sent-events to stream the response
            # print(f"Streaming token: {token}")  # 여기에 print 문 추가
            yield f"{token}"

        await task

    return StreamingResponse(send_message(_req.message), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(host="0.0.0.0", 
                port= int(os.getenv("CHATBOT_PORT", "8000")), 
                app=app)