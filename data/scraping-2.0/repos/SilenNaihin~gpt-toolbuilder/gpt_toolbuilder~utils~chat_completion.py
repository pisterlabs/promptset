import openai
from typing import Optional

from .types import MessageDict, ResponseDict


def create_chat_completion(
    messages: list[MessageDict],
    model: str,
    temperature: Optional[float],
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = 1,
    n: Optional[int] = 1,
    stop: Optional[str] = None,
    presence_penalty: Optional[float] = 0,
    frequency_penalty: Optional[float] = 0,
    functions: Optional[dict[str, str]] = None,
    function_call: Optional[dict[str, str]] = None,
) -> ResponseDict:
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "n": n,
        "stop": stop,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
    }

    response = openai.ChatCompletion.create(**kwargs)

    return ResponseDict(response)  # type: ignore
