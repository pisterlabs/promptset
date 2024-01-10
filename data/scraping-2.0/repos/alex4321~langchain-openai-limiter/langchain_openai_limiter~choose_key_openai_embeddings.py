"""
Wrapper to choose between a few OpenAI keys before embeddings
"""
import copy
from typing import Union, List
from langchain.embeddings.base import Embeddings
from langchain.embeddings.openai import OpenAIEmbeddings
import tiktoken
from .limit_info import choose_key, achoose_key, ApiKey
from .limit_await_openai_embeddings import LimitAwaitOpenAIEmbeddings


_LIMIT_AWAIT_SLEEP = 0.01
_LIMIT_AWAIT_TIMEOUT = 60.0


class ChooseKeyOpenAIEmbeddings(Embeddings):
    """
    Key-choosing OpenAI embeddings wrapper
    """
    def __init__(self, openai_embeddings: Union[LimitAwaitOpenAIEmbeddings, OpenAIEmbeddings],
                 openai_api_keys: List[ApiKey],
                 limit_await_timeout: float = _LIMIT_AWAIT_TIMEOUT,
                 limit_await_sleep: float = _LIMIT_AWAIT_SLEEP):
        super().__init__()
        self.openai_embeddings = openai_embeddings
        self.openai_api_keys = openai_api_keys
        self.limit_await_timeout = limit_await_timeout
        self.limit_await_sleep = limit_await_sleep
    
    @property
    def model(self) -> str:
        return self.openai_embeddings.model

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
        openai_embeddings = copy.deepcopy(self.openai_embeddings)
        openai_embeddings.openai_api_key = choose_key(
            self.openai_embeddings.model,
            self.openai_api_keys,
            token_count
        )
        return openai_embeddings.embed_documents(texts)

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
        openai_embeddings = copy.deepcopy(self.openai_embeddings)
        openai_embeddings.openai_api_key = await achoose_key(
            self.openai_embeddings.model,
            self.openai_api_keys,
            token_count
        )
        return await openai_embeddings.aembed_documents(texts)

    async def aembed_query(self, text: str) -> List[float]:
        """
        Get query embeddings
        """
        return (await self.aembed_documents([text]))[0]
