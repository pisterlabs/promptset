from langchain.callbacks.manager import (CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun)
from langchain.llms.base import create_base_retry_decorator
from langchain.llms.base import BaseLLM
from langchain.chat_models.base import BaseChatModel
from http import HTTPStatus
import asyncio

import logging

from typing import Optional, Union, Callable, Any, AsyncGenerator

logger = logging.getLogger(__name__)


def completion_with_retry(
    llm_model: BaseLLM | BaseChatModel,
    run_manager: Optional[CallbackManagerForLLMRun] = None,
    **kwargs: Any
) -> Any:
    """Use tenacity to retry the completion call."""
    retry_decorator = _create_retry_decorator(llm_model, run_manager=run_manager)

    @retry_decorator
    def _completion_with_retry(**_kwargs: Any) -> Any:
        print("#" * 60)
        print("kwargs: ", _kwargs)

        resp = llm_model.client.call(**_kwargs)
        return resp

    return _completion_with_retry(**kwargs)


# 注意 该函数返回的类型为 <async_generator>
# 仅用在 streaming 调用的 async for 循环中
async def acompletion_with_retry(
    llm_model: BaseLLM | BaseChatModel,
    run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
    **kwargs: Any
) -> AsyncGenerator:
    """Use tenacity to retry the completion call."""
    retry_decorator = _create_retry_decorator(llm_model, run_manager=run_manager)

    @retry_decorator
    async def _completion_with_retry(**_kwargs: Any) -> AsyncGenerator:
        print("#" * 60)
        print("kwargs: ", _kwargs)
        resp = llm_model.client.call(**kwargs)
        print("<<- async resp: ", resp)
        return async_generator(resp)

    return await _completion_with_retry(**kwargs)


async def async_generator(normal_generator):
    for v in normal_generator:
        if v.status_code == HTTPStatus.OK:
            await asyncio.sleep(0)
            yield v
        else:
            print("async_generator HTTP-Err: ", v)
            raise SystemError("http response Error: ", v.status_code)


def _create_retry_decorator(
    llm_model: BaseLLM | BaseChatModel,
    run_manager: Optional[
        Union[AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun]
    ] = None,
) -> Callable[[Any], Any]:
    import dashscope

    errors = [
        # TODO: add more errors
        dashscope.common.error.RequestFailure,
        dashscope.common.error.InvalidInput,
        dashscope.common.error.ModelRequired,
    ]

    return create_base_retry_decorator(
        error_types=errors, max_retries=llm_model.max_retries, run_manager=run_manager
    )


def response_text_format(stream_resp, cursor):
    text = stream_resp["output"]["choices"][0]["message"]["content"]
    text = text[cursor:]
    cursor += len(text)
    stream_resp["output"]["choices"][0]["message"]["content"] = text
    return stream_resp, cursor


def response_handler(response):
    if response.status_code == HTTPStatus.BAD_REQUEST and "contain inappropriate content" in response.message:
        response.status_code = HTTPStatus.OK
        response.output = {
            "choices": [{"finish_reason": "stop", "message": {
                "role": "assistant", "content": "Input data may contain inappropriate content.🐶"}}]
        }
        response.usage = {"output_tokens": 0, "input_tokens": 0}
    elif response.status_code != HTTPStatus.OK:
        raise ValueError(f"http request failed, code: {response.status_code}")
    return response
