import json
import os
from typing import Any, Dict, Optional
import asyncio

import numpy as np
import openai
from retry import retry

from minichain.dtypes import AssistantMessage, FunctionCall
from minichain.message_handler import StreamCollector
from minichain.utils.debug import debug
from minichain.utils.disk_cache import async_disk_cache, disk_cache


def parse_function_call(function_call: Optional[Dict[str, Any]]):
    if function_call is None or function_call.get("name") is None:
        return {}
    try:
        function_call["arguments"] = json.loads(function_call["arguments"])
        return FunctionCall(**function_call)
    except:
        raise Exception(f"Could not parse function call: {function_call}")
    

def fix_common_errors(response: Dict[str, Any]) -> Dict[str, Any]:
    """Fix common errors in the formatting and turn the dict into a AssistantMessage"""
    response["function_call"] = parse_function_call(response["function_call"])
    return response


def format_history(messages: list) -> list:
    """Format the history to be compatible with the openai api - json dumps all arguments"""
    for i, message in enumerate(messages):
        if (function_call := message.get("function_call")) is not None:
            if function_call.get("arguments", None) is not None and isinstance(function_call["arguments"], dict):
                content = function_call["arguments"].pop("content", None)
                message["content"] = content or message["content"]
                function_call["arguments"] = json.dumps(function_call["arguments"])
                message["function_call"] = function_call
            if message['role'] == 'user':
                function_call = message.pop("function_call")
                message['content'] += f"\n**Calling function: {function_call['name']}** with arguments: \n{function_call['arguments']}\n"
        if message['role'] == 'user' or message.get('function_call') is None or message['function_call'].get('name') is None:
            try:
                message.pop("function_call", None)
            except KeyError:
                pass
    return messages


def save_llm_call_for_debugging(messages, functions, parsed_response, raw_response):
    os.makedirs(".minichain/debug", exist_ok=True)
    with open(".minichain/debug/last_openai_request.json", "w") as f:
        json.dump(
            {
                "messages": messages,
                "functions": functions,
                "parsed_response": parsed_response,
                "raw_response": raw_response,
            },
            f,
        )


@async_disk_cache
@retry(tries=10, delay=2, backoff=2, jitter=(1, 3))
async def get_openai_response_stream(
    chat_history, functions, model="gpt-4-0613", stream=None, force_call=None
) -> str:  # "gpt-4-0613", "gpt-3.5-turbo-16k"
    if stream is None:
        stream = StreamCollector()
    messages = format_history(chat_history)

    save_llm_call_for_debugging(
        messages, functions, None, None
    )

    if force_call is not None:
        force_call = {"name": force_call}
    else:
        force_call = "auto"

    try:
        if len(functions) > 0:
            openai_response = await openai.ChatCompletion.acreate(
                model=model,
                messages=messages,
                functions=functions,
                temperature=0.1,
                stream=True,
                function_call=force_call
            )
        else:
            openai_response = await openai.ChatCompletion.acreate(
                model=model,
                messages=messages,
                temperature=0.1,
                stream=True,
                function_call=force_call
            )

        # iterate through the stream of events
        async for chunk in openai_response:
            chunk = chunk["choices"][0]["delta"].to_dict_recursive()
            await stream.chunk(chunk)
    except Exception as e:
        print("We probably got rate limited, chilling for a minute...", e)
        await asyncio.sleep(60)
        raise e
    raw_response = {
        key: value for key, value in stream.current_message.items() if "id" not in key
    }
    response = fix_common_errors(raw_response)
    await stream.set(response)
    save_llm_call_for_debugging(
        messages, functions, response, raw_response=raw_response
    )
    return response


@disk_cache
@retry(tries=10, delay=2, backoff=2, jitter=(1, 3))
@debug
def get_embedding(text):
    response = openai.Embedding.create(model="text-embedding-ada-002", input=text)
    return np.array(response["data"][0]["embedding"])
