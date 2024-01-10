"""
Module for rate/token per minute waiting OpenAIEmbeddings wrapper
"""
from typing import List
from langchain.embeddings.base import Embeddings
from langchain.embeddings.openai import OpenAIEmbeddings
import tiktoken
from .limit_info import wait_for_limit, await_for_limit
from .capture_headers import attach_session_hooks


_LIMIT_AWAIT_SLEEP = 0.01
_LIMIT_AWAIT_TIMEOUT = 60.0


class LimitAwaitOpenAIEmbeddings(Embeddings):
    """
    Rate/Token Per Minute waiting OpenAIEmbeddings wrapper
    """
    def __init__(self, openai_embeddings: OpenAIEmbeddings,
                 limit_await_timeout: float = _LIMIT_AWAIT_TIMEOUT,
                 limit_await_sleep: float = _LIMIT_AWAIT_SLEEP):
        super().__init__()
        self.openai_embeddings = openai_embeddings
        self.limit_await_timeout = limit_await_timeout
        self.limit_await_sleep = limit_await_sleep

    @property
    def openai_api_key(self) -> str:
        """
        Get OpenAI api key
        """
        return self.openai_embeddings.openai_api_key

    @openai_api_key.setter
    def openai_api_key(self, key: str) -> None:
        """
        Set OpenAI api key
        """
        self.openai_embeddings.openai_api_key = key

    @property
    def model(self):
        return self.openai_embeddings.model

    @property
    def model(self) -> str:
        return self.openai_embeddings.model
    
    @model.setter
    def model(self, value: str) -> None:
        self.openai_embeddings.model = value

    def get_num_tokens(self, texts: List[str]) -> int:
        """
        Count tokens in texts
        """
        encoding = tiktoken.encoding_for_model(self.openai_embeddings.model)
        token_ids = encoding.encode_batch(texts)
        total_length = 0
        for row in token_ids:
            total_length += len(row)
        return total_length

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Get document embeddings
        """
        token_count = self.get_num_tokens(texts)
        wait_for_limit(
            self.openai_embeddings.model,
            self.openai_api_key,
            token_count,
            self.limit_await_timeout,
            self.limit_await_sleep,
        )
        return self.openai_embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """
        Get query embeddings
        """
        return self.embed_documents([text])[0]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Get document embeddings
        """
        token_count = self.get_num_tokens(texts)
        if not self.openai_embeddings.headers:
            self.openai_embeddings.headers = {}
        self.openai_embeddings.headers["x-model"] = self.openai_embeddings.model
        await await_for_limit(
            self.openai_embeddings.model,
            self.openai_api_key,
            token_count,
            self.limit_await_timeout,
            self.limit_await_sleep,
        )
        return await self.openai_embeddings.aembed_documents(texts)

    async def aembed_query(self, text: str) -> List[float]:
        """
        Get query embeddings
        """
        return (await self.aembed_documents([text]))[0]


attach_session_hooks()
