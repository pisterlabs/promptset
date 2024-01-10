import asyncio
import copy
import logging
import time
from enum import Enum

import json_repair
import openai
import tiktoken
from httpx import Timeout
from openai import AsyncStream
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionNamedToolChoiceParam
from openai.types.edit import Choice
import streamlit as st

from common.config import settings
from common.context_vars import total_stats

encoder = tiktoken.encoding_for_model('gpt-4')


class TokenCountTooHigh(ValueError):
    pass


MODEL_NAME, VERSION = 'glix', "2023-12-01-preview"

_openai_client = openai.AsyncOpenAI(api_key=settings.OPENAI_KEY)
_azure_client = openai.AsyncAzureOpenAI(
    api_key=settings.AZURE_OPENAI_KEY,
    api_version=VERSION,
    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
    azure_deployment=MODEL_NAME,
)
_azure_client1 = openai.AsyncAzureOpenAI(
    api_key=settings.OPENAI_BACKUP_KEY,
    api_version=VERSION,
    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT_BACKUP,
    azure_deployment=MODEL_NAME,
)


def report_stats(openai_resp: CompletionUsage):
    stats = openai_resp.model_dump()
    val = copy.deepcopy(total_stats.get())
    val.update(stats)
    logging.info('token cost updated stats: %s', val)
    total_stats.set(val)


class Model(Enum):
    GPT4V = (_azure_client, MODEL_NAME)
    GPT4B = (_azure_client1, MODEL_NAME)
    GPT4_32K = (_openai_client, 'gpt-4-32k-0613')


async def send_request(
        messages: list[dict[str, str]],
        seed: int,
        model: Model,
        max_tokens: int | None = None,
        top_p: float | None = None,
        temperature: float | None = None,
        num_options: int | None = None,
        retry_count: int = 1,
        **kwargs
) -> list[Choice] | ChatCompletion | AsyncStream[ChatCompletionChunk]:
    """
    :param messages: list of messages to send to openai
    :param seed: a fixed int to pass to the model. This helps make the model deterministic.
    :param model: the model to use, one of the Model enum values
    :param max_tokens: the maximum number of tokens to generate, None for unlimited, not including the prompt.
    :param top_p: the cumulative probability of the most likely tokens to use.
    :param temperature: the temperature to use, higher means more random.
    :param num_options: the number of options to generate, only used for completion.
    :param retry_count: the number of times to retry the request if it times out.
    :param kwargs: additional parameters to pass to the openai request.
    """

    client, model_name = model.value

    req = dict(messages=messages, seed=seed, model=model_name, max_tokens=max_tokens)
    if 'functions' not in kwargs:
        req['response_format'] = {"type": "json_object"}
    else:
        req['function_call'] = {'name': kwargs['functions'][0]['name']}
    if top_p:
        req['top_p'] = top_p
    if temperature:
        req['temperature'] = temperature
    if num_options:
        req['n'] = num_options
    if kwargs:
        req.update(kwargs)

    try:
        t1 = time.time()
        func_resp = await client.chat.completions.create(**req)
        t2 = time.time()
        logging.info('finished openai request, took %s seconds', t2 - t1)
        # report_stats(func_resp)
        return func_resp
    except openai.APITimeoutError as e:
        logging.exception('openai timeout error, sleeping for 5 seconds and retrying')
        st.toast('OpenAI is taking too long to respond, please wait...')
        await asyncio.sleep(5)
        if retry_count == 3:
            logging.exception('openai timeout error, failed after 3 retries')
            raise e
        return await send_request(messages=messages, seed=seed, model=model, temperature=temperature, top_p=top_p,
                                  retry_count=retry_count + 1, num_options=num_options, max_tokens=max_tokens)
