from __future__ import annotations
from typing import *

import logging

logger = logging.getLogger(__name__)

import json

import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential

from cogniq.config import OPENAI_CHAT_MODEL, OPENAI_MAX_TOKENS_RESPONSE, OPENAI_API_KEY

from .summarizer import Summarizer


class CogniqOpenAI:
    CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"
    COMPLETIONS_URL = "https://api.openai.com/v1/completions"
    API_KEY = OPENAI_API_KEY

    def __init__(self):
        """
        OpenAI model

        """
        # initialize summarizer
        self.summarizer = Summarizer(
            async_chat_completion_create=self.async_chat_completion_create,
        )

    async def async_chat_completion_create(
        self, *, messages: List[Dict[str, str]], stream_callback: Callable[..., None] | None = None, **kwargs
    ) -> Dict[str, Any]:
        stream_callback_set = stream_callback is not None
        url = self.CHAT_COMPLETIONS_URL
        default_payload = {
            "model": OPENAI_CHAT_MODEL,
            "messages": messages,
            "stream": stream_callback_set,
            "max_tokens": OPENAI_MAX_TOKENS_RESPONSE,
        }
        payload = {**default_payload, **kwargs}  # add and override any additional kwargs to payload

        if stream_callback_set:
            return await self.async_openai_stream(url=url, payload=payload, stream_callback=stream_callback, **kwargs)  # type: ignore # since mypy is not picking up on the control flow that ensures stream_callback is not None
        else:
            return await self.async_openai(url=url, payload=payload, **kwargs)

    async def async_completion_create(self, *, prompt: str, **kwargs) -> Dict[str, Any]:
        url = self.COMPLETIONS_URL
        payload = {"prompt": prompt, **kwargs}

        return await self.async_openai(url=url, payload=payload, **kwargs)

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=60))
    async def async_openai(self, *, url: str, payload: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.API_KEY}",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise Exception(f"Error {response.status}: {await response.text()}")

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=60))
    async def async_openai_stream(
        self, *, url: str, payload: Dict[str, Any], stream_callback: Callable[..., None], **kwargs
    ) -> Dict[str, Any]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.API_KEY}",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status == 200:
                    # Tokens will be sent as data-only server-sent events as they become available,
                    # with the stream terminated by a data: [DONE] message.
                    final_content = {"choices": [{"message": {"content": ""}}]}
                    while True:
                        line = await response.content.readline()
                        line = line.strip()
                        if line == b"data: [DONE]":
                            return final_content
                        elif line.startswith(b"data: "):
                            line = line[len(b"data: ") :]
                            obj = json.loads(line.decode("utf-8"))
                            try:
                                delta = obj.get("choices", [{}])[0].get("delta", {})
                                content = delta.get("content")
                                if content:
                                    final_content["choices"][0]["message"]["content"] += content
                                    stream_callback(content)
                            except (KeyError, IndexError):
                                logger.error("Unexpected data structure: %s", obj)
                else:
                    raise Exception(f"Error {response.status}: {await response.text()}")
