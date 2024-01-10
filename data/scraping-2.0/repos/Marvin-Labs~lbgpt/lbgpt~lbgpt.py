# -*- coding: utf-8 -*-
import datetime
from asyncio import Timeout
from logging import getLogger
from typing import Any, Optional, Sequence

import httpx
import openai
from openai._types import NOT_GIVEN, NotGiven

from lbgpt.allocation import (
    max_headroom_allocation_function,
    random_allocation_function,
)
from lbgpt.base import _BaseGPT
from lbgpt.types import ChatCompletionAddition
from lbgpt.usage import Usage

logger = getLogger(__name__)


class ChatGPT(_BaseGPT):
    def __init__(
        self,
        api_key: str,
        max_parallel_calls: int = 5,
        request_timeout: float | Timeout | None | NotGiven = NOT_GIVEN,
        cache: Optional[Any] = None,
        semantic_cache: Optional[Any] = None,
        propagate_standard_cache_to_semantic_cache: bool = False,
        stop_after_attempts: Optional[int] = 10,
        stop_on_exception: bool = False,
        max_usage_cache_size: Optional[int] = 1_000,
        limit_tpm: Optional[int] = None,
        limit_rpm: Optional[int] = None,
    ):
        super().__init__(
            cache=cache,
            semantic_cache=semantic_cache,
            propagate_standard_cache_to_semantic_cache=propagate_standard_cache_to_semantic_cache,
            max_parallel_calls=max_parallel_calls,
            stop_after_attempts=stop_after_attempts,
            stop_on_exception=stop_on_exception,
            max_usage_cache_size=max_usage_cache_size,
            limit_tpm=limit_tpm,
            limit_rpm=limit_rpm,
        )

        self.api_key = api_key
        self.request_timeout = request_timeout

    def get_client(self) -> openai.AsyncOpenAI:
        return openai.AsyncOpenAI(
            api_key=self.api_key,
            timeout=self.request_timeout,
            max_retries=0,
        )

    async def chat_completion(self, **kwargs) -> ChatCompletionAddition:
        # one request to the OpenAI API respecting their ratelimit

        timeout = kwargs.pop("request_timeout", self.request_timeout)

        start = datetime.datetime.now()
        out = await self.get_client().with_options(
            timeout=timeout
        ).chat.completions.create(
            **kwargs,
        )
        self.add_usage_to_usage_cache(
            Usage(
                input_tokens=out.usage.prompt_tokens,
                output_tokens=out.usage.total_tokens,
                start_datetime=start,
                end_datetime=datetime.datetime.now(),
            )
        )

        return ChatCompletionAddition.from_chat_completion(out)


class AzureGPT(_BaseGPT):
    def __init__(
        self,
        api_key: str,
        azure_api_base: str,
        azure_model_map: dict[str, str],
        cache: Optional[Any] = None,
        semantic_cache: Optional[Any] = None,
        propagate_standard_cache_to_semantic_cache: bool = False,
        azure_openai_version: str = "2023-05-15",
        azure_openai_type: str = "azure",
        max_parallel_calls: int = 5,
        request_timeout: float | Timeout | None | NotGiven = NOT_GIVEN,
        stop_after_attempts: Optional[int] = 10,
        stop_on_exception: bool = False,
        max_usage_cache_size: Optional[int] = 1_000,
        limit_tpm: Optional[int] = None,
        limit_rpm: Optional[int] = None,
    ):
        super().__init__(
            cache=cache,
            semantic_cache=semantic_cache,
            propagate_standard_cache_to_semantic_cache=propagate_standard_cache_to_semantic_cache,
            max_parallel_calls=max_parallel_calls,
            stop_after_attempts=stop_after_attempts,
            stop_on_exception=stop_on_exception,
            max_usage_cache_size=max_usage_cache_size,
            limit_tpm=limit_tpm,
            limit_rpm=limit_rpm,
        )

        self.api_key=api_key
        self.azure_api_base=azure_api_base
        self.azure_openai_version=azure_openai_version
        self.request_timeout=request_timeout

        self.azure_model_map = azure_model_map

    def get_client(self) -> openai.AsyncAzureOpenAI:
        return openai.AsyncAzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=self.azure_api_base,
            api_version=self.azure_openai_version,
            timeout=self.request_timeout,
            max_retries=0,
        )

    async def chat_completion(self, **kwargs) -> ChatCompletionAddition:
        """One request to the Azure OpenAI API respecting their ratelimit
        # needs to change the model parameter to deployment id
        """

        deployment_id = self.azure_model_map[kwargs["model"]]
        kwargs["model"] = deployment_id

        timeout = kwargs.pop("request_timeout", self.request_timeout)

        start = datetime.datetime.now()
        out = await self.get_client().with_options(
            timeout=timeout
        ).chat.completions.create(
            **kwargs,
        )
        self.add_usage_to_usage_cache(
            Usage(
                input_tokens=out.usage.prompt_tokens,
                output_tokens=out.usage.total_tokens,
                start_datetime=start,
                end_datetime=datetime.datetime.now(),
            )
        )

        return ChatCompletionAddition.from_chat_completion(out)


ALLOCATION_FUNCTIONS = {
    "random": random_allocation_function,
    "max_headroom": max_headroom_allocation_function,
}


class MultiLoadBalancedGPT(_BaseGPT):
    def __init__(
        self,
        gpts: list[_BaseGPT],
        allocation_function: str = "random",
        allocation_function_kwargs: Optional[dict] = None,
        allocation_function_weights: Optional[Sequence] = None,
        cache: Optional[Any] = None,
        semantic_cache: Optional[Any] = None,
        propagate_standard_cache_to_semantic_cache: bool = False,
        stop_after_attempts: Optional[int] = 10,
        stop_on_exception: bool = False,
        max_parallel_requests: Optional[int] = None,
    ):
        self.gpts = gpts

        if isinstance(allocation_function, str):
            allocation_function = ALLOCATION_FUNCTIONS[allocation_function]
        else:
            raise NotImplementedError(
                f"Cannot infer allocation function from type {type(allocation_function)}"
            )

        self.allocation_function = allocation_function
        self.allocation_function_kwargs = allocation_function_kwargs or {}

        if allocation_function_weights is not None:
            assert len(allocation_function_weights) == len(
                gpts
            ), "if provided, `allocation_function_weights` must be the same length as gpts"

        self.allocation_function_weights = allocation_function_weights
        if max_parallel_requests is None:
            max_parallel_requests = sum([gpt.max_parallel_calls for gpt in gpts])

        super().__init__(
            cache=cache,
            semantic_cache=semantic_cache,
            propagate_standard_cache_to_semantic_cache=propagate_standard_cache_to_semantic_cache,
            max_parallel_calls=max_parallel_requests,
            stop_after_attempts=stop_after_attempts,
            stop_on_exception=stop_on_exception,
        )

    @property
    def usage_cache_list(self) -> list[Usage]:
        out = sum([gpt.usage_cache_list for gpt in self.gpts], [])
        return out

    async def chat_completion(self, **kwargs) -> ChatCompletionAddition:
        gpt = await self.allocation_function(
            self.gpts,
            weights=self.allocation_function_weights,
            **self.allocation_function_kwargs,
        )

        return await gpt.chat_completion(**kwargs)


class LoadBalancedGPT(MultiLoadBalancedGPT):
    """
    We are continuing to support this for backward compatability reasons, but it is discouraged to use it.
    """

    def __init__(
        self,
        openai_api_key: str,
        azure_api_key: str,
        azure_api_base: str,
        azure_model_map: dict[str, str],
        cache: Optional[Any] = None,
        semantic_cache: Optional[Any] = None,
        propagate_standard_cache_to_semantic_cache: bool = False,
        azure_openai_version: str = "2023-05-15",
        azure_openai_type: str = "azure",
        max_parallel_calls_openai: int = 5,
        max_parallel_calls_azure: int = 5,
        ratio_openai_to_azure: float = 0.25,
        stop_after_attempts: Optional[int] = 10,
        stop_on_exception: bool = False,
    ):
        self.openai = ChatGPT(
            api_key=openai_api_key,
            cache=cache,
            semantic_cache=semantic_cache,
            propagate_standard_cache_to_semantic_cache=propagate_standard_cache_to_semantic_cache,
            max_parallel_calls=max_parallel_calls_openai,
            stop_after_attempts=stop_after_attempts,
            stop_on_exception=stop_on_exception,
        )

        self.azure = AzureGPT(
            api_key=azure_api_key,
            azure_api_base=azure_api_base,
            azure_model_map=azure_model_map,
            azure_openai_version=azure_openai_version,
            azure_openai_type=azure_openai_type,
            cache=cache,
            semantic_cache=semantic_cache,
            propagate_standard_cache_to_semantic_cache=propagate_standard_cache_to_semantic_cache,
            max_parallel_calls=max_parallel_calls_azure,
            stop_after_attempts=stop_after_attempts,
            stop_on_exception=stop_on_exception,
        )

        super().__init__(
            gpts=[self.openai, self.azure],
            cache=cache,
            semantic_cache=semantic_cache,
            propagate_standard_cache_to_semantic_cache=propagate_standard_cache_to_semantic_cache,
            allocation_function="random",
            allocation_function_weights=[
                ratio_openai_to_azure,
                1 - ratio_openai_to_azure,
            ],
            stop_after_attempts=stop_after_attempts,
            stop_on_exception=stop_on_exception,
        )
