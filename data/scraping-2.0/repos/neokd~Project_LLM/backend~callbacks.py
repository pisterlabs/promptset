from langchain.callbacks.base import BaseCallbackHandler,AsyncCallbackHandler
from typing import Any, Generator, Optional
from threading import Thread
from queue import Queue, Empty
from langchain.globals import get_llm_cache
from sse_starlette.sse import ServerSentEvent
from sse_starlette.sse import ensure_bytes as ensure_bytes
from starlette.types import Send, Message
from enum import Enum
from pydantic import BaseModel
import pydantic
from langchain.docstore.document import Document
from fastapi import status
from sse_starlette.sse import EventSourceResponse
import asyncio
from functools import partial
from langchain.chains.base import Chain

################### Callbacks for /api/chat ###################################

PYDANTIC_V2 = pydantic.VERSION.startswith("2.")
class StrEnum(str, Enum):
    ...

class TokenStreamMode(StrEnum):
    TEXT = "text"
    JSON = "json"

class LangchainEvents(StrEnum):
    SOURCE_DOCUMENTS = "source_documents"

class TokenEventData(BaseModel):
    token: str = ""

class Events(StrEnum):
    COMPLETION = "completion"
    ERROR = "error"
    END = "end"

class CallbackHandler(AsyncCallbackHandler):
    """Callback handler that prints the output to stdout."""
    def __init__(self, **kwargs: dict[str, Any]) -> None:
        super().__init__(**kwargs)
        self.llm_cache_used = get_llm_cache() is not None

    @property
    def always_verbose(self) -> bool:
        return True
    
class ChainRunMode(StrEnum):
    """Enum for LangChain run modes."""
    ASYNC = "async"
    SYNC = "sync"


class HTTPStatusDetail(StrEnum):
    INTERNAL_SERVER_ERROR = "Internal Server Error"

def model_dump_json(model: BaseModel, **kwargs) -> str:
    if PYDANTIC_V2:
        return model.model_dump_json(**kwargs)
    else:
        return model.json(**kwargs)


def get_token_data(token: str, mode: TokenStreamMode) -> str:
    if mode not in list(TokenStreamMode):
        raise ValueError(f"Invalid stream mode: {mode}")

    if mode == TokenStreamMode.TEXT:
        return token
    else:
        return model_dump_json(TokenEventData(token=token))  

class StreamingCallbackHandler(CallbackHandler):

    def __init__( self,*,send: Send = None, **kwargs: dict[str, Any],) -> None:
        super().__init__(**kwargs)
        self._send = send
        self.streaming = None

    @property
    def send(self) -> Send:
        return self._send

    @send.setter
    def send(self, value: Send) -> None:
        if not callable(value):
            raise ValueError("value must be a Callable")
        self._send = value

    def _construct_message(self, data: str, event: Optional[str] = None) -> Message:
        chunk = ServerSentEvent(data=data, event=event)
        return {
            "type": "http.response.body",
            "body": ensure_bytes(chunk, None),
            "more_body": True,
        }

class TokenStreamingCallbackHandler(StreamingCallbackHandler):
    """Callback handler for streaming tokens."""

    def __init__(self,*,output_key: str,mode: TokenStreamMode = TokenStreamMode.JSON,**kwargs: dict[str, Any],) -> None:
        super().__init__(**kwargs)

        self.output_key = output_key

        if mode not in list(TokenStreamMode):
            raise ValueError(f"Invalid stream mode: {mode}")
        self.mode = mode

    async def on_chain_start(self, *args: Any, **kwargs: dict[str, Any]) -> None:
        self.streaming = False

    async def on_llm_new_token(self, token: str, **kwargs: dict[str, Any]) -> None:
        if not self.streaming:
            self.streaming = True

        if self.llm_cache_used:  # cache missed (or was never enabled) if we are here
            self.llm_cache_used = False

        message = self._construct_message(
            data=get_token_data(token, self.mode), event=Events.COMPLETION
        )
        await self.send(message)

    async def on_chain_end(
        self, outputs: dict[str, Any], **kwargs: dict[str, Any]
    ) -> None:
        if self.llm_cache_used or not self.streaming:
            if self.output_key in outputs:
                message = self._construct_message(
                    data=get_token_data(outputs[self.output_key], self.mode),
                    event=Events.COMPLETION,
                )
                await self.send(message)
            else:
                raise KeyError(f"missing outputs key: {self.output_key}")

class SourceDocumentsEventData(BaseModel):
    source_documents: list[dict[str, Any]]

class SourceDocumentsStreamingCallbackHandler(StreamingCallbackHandler):
    async def on_chain_end(self, outputs: dict[str, Any], **kwargs: dict[str, Any]) -> None:
        if "source_documents" in outputs:
            if not isinstance(outputs["source_documents"], list):
                raise ValueError("source_documents must be a list")
            if not isinstance(outputs["source_documents"][0], Document):
                raise ValueError("source_documents must be a list of Document")

            # NOTE: langchain is using pydantic_v1 for `Document`
            source_documents: list[dict] = [
                document.dict() for document in outputs["source_documents"]
            ]
            message = self._construct_message(
                data=model_dump_json(
                    SourceDocumentsEventData(source_documents=source_documents)
                ),
                event=LangchainEvents.SOURCE_DOCUMENTS,
            )
            await self.send(message)


class _StreamingResponse(EventSourceResponse):

    def __init__(self,content: Any = iter(()),*args: Any,**kwargs: dict[str, Any],) -> None:

        super().__init__(content=content, *args, **kwargs)

    async def stream_response(self, send: Send) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": self.status_code,
                "headers": self.raw_headers,
            }
        )

        try:
            async for data in self.body_iterator:
                chunk = ensure_bytes(data, self.sep)
                await send({"type": "http.response.body", "body": chunk, "more_body": True})
        except Exception as e:
            chunk = ServerSentEvent(
                data=dict(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=HTTPStatusDetail.INTERNAL_SERVER_ERROR,
                ),
                event=Events.ERROR,
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": ensure_bytes(chunk, None),
                    "more_body": True,
                }
            )

        await send({"type": "http.response.body", "body": b"", "more_body": False})

class StreamingResponse(_StreamingResponse):
    """StreamingResponse class for LangChain resources."""

    def __init__(self,chain: Chain,config: dict[str, Any],run_mode: ChainRunMode = ChainRunMode.ASYNC, *args: Any,**kwargs: dict[str, Any],) -> None:
        super().__init__(*args, **kwargs)

        self.chain = chain
        self.config = config

        if run_mode not in list(ChainRunMode):
            raise ValueError(
                f"Invalid run mode '{run_mode}'. Must be one of {list(ChainRunMode)}"
            )

        self.run_mode = run_mode

    async def stream_response(self, send: Send) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": self.status_code,
                "headers": self.raw_headers,
            }
        )

        if "callbacks" in self.config:
            for callback in self.config["callbacks"]:
                if hasattr(callback, "send"):
                    callback.send = send

        try:
            if self.run_mode == ChainRunMode.ASYNC:
                outputs = await self.chain.acall(**self.config)
            else:
                loop = asyncio.get_event_loop()
                outputs = await loop.run_in_executor(
                    None, partial(self.chain, **self.config)
                )
            if self.background is not None:
                self.background.kwargs.update({"outputs": outputs})
        except Exception as e:
            if self.background is not None:
                self.background.kwargs.update({"outputs": {}, "error": e})
            chunk = ServerSentEvent(
                data=dict(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=HTTPStatusDetail.INTERNAL_SERVER_ERROR,
                ),
                event=Events.ERROR,
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": ensure_bytes(chunk, None),
                    "more_body": True,
                }
            )

        await send({"type": "http.response.body", "body": b"", "more_body": False})

########################### Callbacks for /api/stream ###########################
class QueueCallbackHandler(BaseCallbackHandler):
    def __init__(self, queue):
        self.queue = queue

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.queue.put(
            {
                "event": "event",
                "data": token,
            }
        )

    def on_llm_end(self, *args, **kwargs) -> Any:
        return self.queue.empty()

def stream(cb: Any, queue: Queue) -> Generator:
    job_done = object()

    def task():
        cb()
        queue.put(job_done)

    t = Thread(target=task)
    t.start()

    while True:
        try:
            item = queue.get(True, timeout=1)
            if item is job_done:
                t.join()  # Wait for the thread to finish
                break
            yield item
        except Empty:
            continue
