import asyncio
import json
from functools import partial
from io import BytesIO
from typing import Union, BinaryIO

import loguru
import openai
import pydub
import tiktoken
from aiolimiter import AsyncLimiter

WHISPER_RATE_LIMIT = 50  # 50 requests per minute
whisper_limiter = AsyncLimiter(WHISPER_RATE_LIMIT, 60)  # 50 requests per minute
GPT_RATE_LIMIT = 200  # 200 requests per minute
gpt_limiter = AsyncLimiter(GPT_RATE_LIMIT, 60)  # 200 requests per minute


# Then use atranscribe_audio_limited instead of atranscribe_audio

token_limit_by_model = {
    "gpt-3.5-turbo": 4096,
    "gpt-4": 8192,
    "gpt-3.5-turbo-16k": 16384,
}


def get_token_count(text, model="gpt-3.5-turbo"):
    """
    calculate amount of tokens in text
    model: gpt-3.5-turbo, gpt-4
    """
    # To get the tokeniser corresponding to a specific model in the OpenAI API:
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))


# todo: add retry in case of error. Or at least handle gracefully
def run_command_with_gpt(command: str, data: str, model="gpt-3.5-turbo"):
    messages = [
        {"role": "system", "content": command},
        {"role": "user", "content": data},
    ]
    response = openai.ChatCompletion.create(messages=messages, model=model)
    return response.choices[0].message.content


# todo: if reason is length - continue generation
async def arun_command_with_gpt(command: str, data: str, model="gpt-3.5-turbo"):
    messages = [
        {"role": "system", "content": command},
        {"role": "user", "content": data},
    ]
    async with gpt_limiter:
        response = await openai.ChatCompletion.acreate(messages=messages, model=model)
    return response.choices[0].message.content


Audio = Union[pydub.AudioSegment, BytesIO, BinaryIO, str]


def transcribe_audio(audio: Audio, model="whisper-1"):
    if isinstance(audio, str):
        audio = open(audio)
    return openai.Audio.transcribe(model, audio).text


async def atranscribe_audio(audio: Audio, model="whisper-1"):
    if isinstance(audio, str):
        audio = open(audio)
    async with whisper_limiter:
        result = await openai.Audio.atranscribe(model, audio)
    return result.text


def default_merger(chunks, keyword="TEMPORARY_RESULT:"):
    return "\n".join([f"{keyword}\n{chunk}" for chunk in chunks])


def split_by_weight(items, weight_func, limit):
    groups = []
    group = []
    group_weight = 0

    for item in items:
        item_weight = weight_func(item)
        if group_weight + item_weight > limit:
            if not group:
                raise ValueError(
                    f"Item {item} is too big to fit into a single group with limit {limit}"
                )
            groups.append(group)
            group = []
            group_weight = 0
        group.append(item)
        group_weight += item_weight

    if group:  # If there are items left in the current group, append it to groups.
        groups.append(group)

    return groups


async def apply_command_recursively(
    command, chunks, model="gpt-3.5-turbo", merger=None, logger=None
):
    """
    Apply GPT command recursively to the data
    """
    if logger is None:
        logger = loguru.logger
    if merger is None:
        merger = default_merger
    token_limit = token_limit_by_model[model]
    while len(chunks) > 1:
        groups = split_by_weight(
            chunks, partial(get_token_count, model=model), token_limit
        )
        if len(groups) == len(chunks):
            raise ValueError(
                f"Chunk size is too big for model {model} with limit {token_limit}"
            )
        logger.debug(f"Split into {len(groups)} groups")
        # apply merger
        merged_chunks = map(merger, groups)
        # apply command
        chunks = await amap_gpt_command(merged_chunks, command, model=model)
        logger.debug(f"Intermediate Result: {chunks}")

    return chunks[0]


def map_gpt_command(
    chunks, command, all_results=False, model="gpt-3.5-turbo", logger=None
):
    """
    Run GPT command on each chunk one by one
    Accumulating temporary results and supplying them to the next chunk
    """
    if logger is None:
        logger = loguru.logger
    logger.debug(f"Running command: {command}")

    temporary_results = None
    results = []
    for chunk in chunks:
        data = {"TEXT": chunk, "TEMPORARY_RESULTS": temporary_results}
        data_str = json.dumps(data, ensure_ascii=False)
        temporary_results = run_command_with_gpt(command, data_str, model=model)
        results.append(temporary_results)

    logger.debug(f"Results: {results}")
    if all_results:
        return results
    else:
        return results[-1]


MERGE_COMMAND_TEMPLATE = """
You're merge assistant. The following command was applied to each chunk.
The results are separated by keyword "{keyword}"
You have to merge all the results into one. 
COMMAND:
{command}
"""


async def amap_gpt_command(chunks, command, model="gpt-3.5-turbo", merge=False):
    """
    Run GPT command on each chunk in parallel
    Merge results if merge=True
    """
    tasks = [arun_command_with_gpt(command, chunk, model=model) for chunk in chunks]

    # Using asyncio.gather to collect all results
    completed_tasks = await asyncio.gather(*tasks)

    if merge:
        merge_command = MERGE_COMMAND_TEMPLATE.format(
            command=command, keyword="TEMPORARY_RESULT:"
        ).strip()
        return apply_command_recursively(merge_command, completed_tasks, model=model)
    else:
        return completed_tasks
