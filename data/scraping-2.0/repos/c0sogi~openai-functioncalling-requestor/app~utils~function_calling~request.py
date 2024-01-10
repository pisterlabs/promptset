from asyncio import wait_for
from typing import Any, Coroutine, Optional

from ...models.completion_models import ChatCompletion, FunctionCallParsed
from ...models.function_calling.base import FunctionCall
from ...utils.api.completion import request_chat_completion
from .parser import make_function_call_parsed_from_dict


async def request_function_call(
    messages: list[dict[str, str]],
    functions: list[FunctionCall],
    function_call: Optional[FunctionCall | str] = "auto",
    model: str = "gpt-3.5-turbo",
    api_base: str = "https://api.openai.com/v1",
    api_key: Optional[str] = None,
    timeout: Optional[float] = None,
    force_arguments: bool = False,
    **kwargs: Any,
) -> FunctionCallParsed:
    """Request a function call from OpenAI's API."""
    coro: Coroutine[Any, Any, ChatCompletion] = request_chat_completion(
        messages=messages,
        model=model,
        api_base=api_base,
        api_key=api_key,
        functions=functions,
        function_call=function_call,
        **kwargs,
    )
    if timeout is not None:
        coro = wait_for(coro, timeout=timeout)
    function_call_unparsed = (await coro)["choices"][0]["message"].get("function_call")
    if function_call_unparsed is None:
        raise ValueError("No function call returned")
    function_call_parsed = make_function_call_parsed_from_dict(function_call_unparsed)
    if force_arguments and "arguments" not in function_call_parsed:
        raise ValueError("No arguments returned")

    return function_call_parsed
