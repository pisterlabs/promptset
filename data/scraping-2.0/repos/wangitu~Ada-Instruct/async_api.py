import sys
import os
from dataclasses import dataclass
from threading import Thread
from typing import Optional, Any
import openai

import asyncio
import aiohttp
from aioprocessing import AioJoinableQueue, AioQueue
from tenacity import wait_random_exponential, stop_after_attempt, AsyncRetrying, RetryError


@dataclass
class Payload:
    data: dict
    metadata: Optional[dict]
    max_retries: int
    retry_multiplier: float
    retry_max: float
    endpoint: str = None
    api_key: str = None
    attempt: int = 0
    failed: bool = False
    response: Any = None
    callback: Any = None

    def call_callback(self):
        if self.callback:
            self.callback(self)
            
            
async def legacy_api(payload: Payload, **kwargs):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {payload.api_key}"
    }
    
    async with aiohttp.ClientSession(headers=headers) as session:
        response = await session.post(
            url="https://api.openai.com/v1/chat/completions",
            json=payload.data,
            **kwargs
        )
        json_response = await response.json()
        
    if "error" in json_response:
        raise ValueError(json_response['error']['message'])

    return json_response
    

class OpenAIMultiClient:
    def __init__(self,
            concurrency: int = 8,
            max_retries: int = 3,
            wait_interval: float = 0,
            retry_multiplier: float = 1,
            retry_max: float = 60,
            data_template: Optional[dict] = None,
            metadata_template: Optional[dict] = None,
            endpoint: Optional[str] = None,
            custom_api=None
    ):
        self._wait_interval = wait_interval
        self._data_template = data_template or {}
        self._metadata_template = metadata_template or {}
        self._max_retries = max_retries
        self._retry_multiplier = retry_multiplier
        self._retry_max = retry_max
        self._concurrency = concurrency
        self._loop = asyncio.new_event_loop()
        self._in_queue = AioJoinableQueue(maxsize=concurrency)
        self._out_queue = AioQueue(maxsize=concurrency)
        self._event_loop_thread = Thread(target=self._run_event_loop)
        self._event_loop_thread.start()
        self._endpoint = endpoint
        self._mock_api = custom_api
        for i in range(concurrency):
            asyncio.run_coroutine_threadsafe(self._worker(i), self._loop)

    def run_request_function(self, input_function, *args, stop_at_end=True, **kwargs):
        if stop_at_end:
            def f(*args, **kwargs):
                input_function(*args, **kwargs)
                self.close()
        else:
            f = input_function
        input_thread = Thread(target=f, args=args, kwargs=kwargs)
        input_thread.start()

    def _run_event_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _process_payload(self, payload: Payload) -> Payload:
        # print(f"Processing {payload.metadata}")
        if self._mock_api:
            payload.response = await self._mock_api(payload)
        elif payload.endpoint == "completions":
            payload.response = await openai.Completion.acreate(**payload.data)
        elif payload.endpoint == "chat.completions" or payload.endpoint == "chats":
            payload.response = await openai.ChatCompletion.acreate(**payload.data)
        elif payload.endpoint == "embeddings":
            payload.response = await openai.Embedding.acreate(**payload.data)
        elif payload.endpoint == "edits":
            payload.response = await openai.Edit.acreate(**payload.data)
        elif payload.endpoint == "images":
            payload.response = await openai.Image.acreate(**payload.data)
        elif payload.endpoint == "fine-tunes":
            payload.response = await openai.FineTune.acreate(**payload.data)
        else:
            raise ValueError(f"Unknown endpoint {payload.endpoint}")
        # print(f"Processed {payload.metadata}")
        return payload

    async def _worker(self, i):
        while True:
            payload = await self._in_queue.coro_get()

            if payload is None:
                print(f"Exiting worker {i}")
                self._in_queue.task_done()
                break

            try:
                async for attempt in AsyncRetrying(
                        wait=wait_random_exponential(multiplier=payload.retry_multiplier, max=payload.retry_max),
                        stop=stop_after_attempt(payload.max_retries)):
                    with attempt:
                        try:
                            payload.attempt = attempt.retry_state.attempt_number
                            payload = await self._process_payload(payload)
                            await self._out_queue.coro_put(payload)
                            self._in_queue.task_done()
                        except:
                            print(f"Error processing {payload.metadata}")
                            raise
            except RetryError as e:
                payload.failed = True
                payload.response = str(e.last_attempt.exception())
                print(f"Failed to process {payload.metadata}: {e.last_attempt.exception()}")
                await self._out_queue.coro_put(payload)
                self._in_queue.task_done()
            await asyncio.sleep(self._wait_interval)

    def close(self):
        try:
            # Putting in sentinel records "null" to tell the queue
            for _ in range(self._concurrency):
                self._in_queue.put(None)
            self._in_queue.join()
            self._out_queue.put(None)
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._event_loop_thread.join()
        except Exception as e:
            print(f"Error closing: {e}")

    def __iter__(self):
        return self

    def __next__(self):
        out = self._out_queue.get()
        if out is None:
            raise StopIteration
        out.call_callback()
        return out

    def request(self,
                data: dict,
                metadata: Optional[dict] = None,
                max_retries: Optional[int] = None,
                retry_multiplier: Optional[float] = None,
                retry_max: Optional[float] = None,
                endpoint: Optional[str] = None,
                api_key: Optional[str] = None,
                callback: Any = None
    ):
        payload = Payload(
            data={**self._data_template, **data},
            metadata={**self._metadata_template, **(metadata or {})},
            max_retries=max_retries or self._max_retries,
            retry_multiplier=retry_multiplier or self._retry_multiplier,
            retry_max=retry_max or self._retry_max,
            endpoint=endpoint or self._endpoint,
            api_key=api_key,
            callback=callback
        )
        self._in_queue.put(payload)

    def pull_all(self):
        for _ in self:
            pass
