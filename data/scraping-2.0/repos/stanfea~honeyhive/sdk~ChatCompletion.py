from typing import Dict, List, Optional, Any
from honeyhive.api.models.generations import (
    GenerationResponse,
    GenerationLoggingQuery,
)
from honeyhive.sdk.init import honeyhive_client

import honeyhive


def create(
    project: str,
    model: str,
    messages: List[str],
    source: Optional[str] = None,
    max_tokens: Optional[int] = 4000,
    temperature: Optional[float] = 1.0,
    top_p: Optional[float] = 1.0,
    suffix: Optional[str] = None,
    presence_penalty: Optional[float] = 0.0,
    frequency_penalty: Optional[float] = 0.0,
    n: Optional[int] = 1,
    stream: Optional[bool] = False,
    logprobs: Optional[Any] = None,
    stop: Optional[str] = None,
    logit_bias: Optional[Dict[str, float]] = None,
    user: Optional[str] = None,
) -> GenerationResponse:
    """Generate completions"""
    # send all the specified args to openai's completion endpoint
    import openai
    import os
    import time

    openai.api_key = os.environ.get("OPENAI_API_KEY", honeyhive.openai_api_key)

    args_for_openai = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "suffix": suffix,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "n": n,
        "stream": stream,
        "logprobs": logprobs,
        "stop": stop,
        "logit_bias": logit_bias,
        "user": user,
    }

    # drop all the args which have None values
    args_for_openai = {
        k: v for k, v in args_for_openai.items() if v is not None
    }

    start = time.time()
    response = openai.ChatCompletion.create(**args_for_openai)
    end = time.time()

    latency = (end - start) * 1000

    # log generation via honeyhive
    client = honeyhive_client()

    # from args for openai, get the args for honeyhive
    # remove the args which are not in honeyhive's logging query
    args_for_honeyhive = args_for_openai.copy()
    args_for_honeyhive.pop("model", None)
    args_for_honeyhive.pop("prompt", None)
    args_for_honeyhive.pop("messages", None)
    # pop user if present
    args_for_honeyhive.pop("user", None)

    # stringify the messages json list
    messages = str(messages)

    honeyhive_response = client.log_generation(
        GenerationLoggingQuery(
            task=project,
            model=model,
            prompt=messages,
            hyperparameters=args_for_honeyhive,
            generation=response.choices[0].message.content,
            usage=response.usage,
            latency=latency,
            source=source,
        )
    )
    generation_id = honeyhive_response.generation_id

    # add generation_id to openai's response
    response.generation_id = generation_id

    return response


__all__ = ["create"]
