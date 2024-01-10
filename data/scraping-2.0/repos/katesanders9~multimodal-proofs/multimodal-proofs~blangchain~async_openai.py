"""OpenAI chat wrapper."""
from __future__ import annotations

import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    List,
    Optional, Union, Dict,
)

from langchain import OpenAI
from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
)
from langchain.chat_models import ChatOpenAI
from langchain.llms.openai import update_token_usage
from langchain.schema import (
    ChatResult,
)
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.messages import (
    BaseMessage,
)
from langchain.schema.output import ChatGenerationChunk, LLMResult, GenerationChunk, Generation
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential, wait_exponential_jitter,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

def _create_retry_decorator(llm: Union[ChatOpenAI, OpenAI]) -> Callable[[Any], Any]:
    import openai

    min_seconds = 4
    max_seconds = 45
    # Wait 2^x * 1 second between each retry starting with
    # 4 seconds, then up to 10 seconds, then 10 seconds afterwards
    return retry(
        reraise=True,
        stop=stop_after_attempt(llm.max_retries),
        wait=wait_exponential_jitter(exp_base=2, initial=min_seconds, max=max_seconds, jitter=5),
        retry=(
            retry_if_exception_type(openai.error.Timeout)
            | retry_if_exception_type(openai.error.APIError)
            | retry_if_exception_type(openai.error.APIConnectionError)
            | retry_if_exception_type(openai.error.RateLimitError)
            | retry_if_exception_type(openai.error.ServiceUnavailableError)
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )


async def acompletion_with_retry(llm: Union[ChatOpenAI, OpenAI], **kwargs: Any) -> Any:
    """Use tenacity to retry the async completion call."""
    retry_decorator = _create_retry_decorator(llm)

    @retry_decorator
    async def _completion_with_retry(**kwargs: Any) -> Any:
        # Use OpenAI's async api https://github.com/openai/openai-python#async-api
        return await llm.client.acreate(**kwargs)

    return await _completion_with_retry(**kwargs)


class JitterWaitOpenAI(OpenAI):
    async def _agenerate(
            self,
            prompts: List[str],
            stop: Optional[List[str]] = None,
            run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> LLMResult:
        """Call out to OpenAI's endpoint async with k unique prompts."""
        params = self._invocation_params
        params = {**params, **kwargs}
        sub_prompts = self.get_sub_prompts(params, prompts, stop)
        choices = []
        token_usage: Dict[str, int] = {}
        # Get the token usage from the response.
        # Includes prompt, completion, and total tokens used.
        _keys = {"completion_tokens", "prompt_tokens", "total_tokens"}
        for _prompts in sub_prompts:
            if self.streaming:
                if len(_prompts) > 1:
                    raise ValueError("Cannot stream results with multiple prompts.")

                generation: Optional[GenerationChunk] = None
                async for chunk in self._astream(
                        _prompts[0], stop, run_manager, **kwargs
                ):
                    if generation is None:
                        generation = chunk
                    else:
                        generation += chunk
                assert generation is not None
                choices.append(
                    {
                        "text": generation.text,
                        "finish_reason": generation.generation_info.get("finish_reason")
                        if generation.generation_info
                        else None,
                        "logprobs": generation.generation_info.get("logprobs")
                        if generation.generation_info
                        else None,
                    }
                )
            else:
                response = await acompletion_with_retry(self, prompt=_prompts, **params)
                choices.extend(response["choices"])
                update_token_usage(_keys, response, token_usage)
        return self.create_llm_result(choices, prompts, token_usage)


class JitterWaitChatOpenAI(ChatOpenAI):
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.streaming:
            generation: Optional[ChatGenerationChunk] = None
            async for chunk in self._astream(
                messages=messages, stop=stop, run_manager=run_manager, **kwargs
            ):
                if generation is None:
                    generation = chunk
                else:
                    generation += chunk
            assert generation is not None
            return ChatResult(generations=[generation])

        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs}
        response = await acompletion_with_retry(self, messages=message_dicts, **params)
        return self._create_chat_result(response)