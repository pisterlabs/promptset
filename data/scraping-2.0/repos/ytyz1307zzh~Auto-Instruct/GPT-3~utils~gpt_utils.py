import pdb
import sys
import time
import openai
import asyncio
import requests
import numpy as np
from typing import List


def print_error(ex: Exception) -> None:
    print('{0}: {1}'.format(ex.__class__.__name__, ex), file=sys.stderr)


async def async_gpt_inference(
    all_async_prompts: List[List[str]],
    model: str,
    max_tokens: int,
    num_return=1,
    best_of=1,
    temperature=0.0,
    top_p=0.0,
    logprobs=None,
    echo=False,
    stop=None,
    sleep=1
):
    completions = []

    for i in range(1, 301):
        try:
            async_responses = [
                openai.Completion.acreate(
                    engine=model,
                    prompt=async_prompts,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    n=num_return,
                    logprobs=logprobs,
                    best_of=best_of,
                    echo=echo,
                    stop=stop
                )
                for async_prompts in all_async_prompts
            ]
            completions = await asyncio.gather(*async_responses)
            break
        except Exception as e:
            if isinstance(e, openai.InvalidRequestError):
                raise e
            print_error(e)
            if i % 100 == 0:
                print(f"tried {i} times, sleep for 10 minutes ...")
                time.sleep(600)
            else:
                print(f"tried {i} times, sleep for {sleep} seconds ...")
                time.sleep(sleep)

    if not completions:
        raise RuntimeError("No responses collected from OpenAI. "
                           "If this comes from rate limits, please lower the number of async coroutines.")

    # Combine all completion["choices"] into one
    combined_completions = []
    for c in completions:
        combined_completions.extend(c["choices"])
    return {"choices": combined_completions}


def gpt_inference(
        inputs_with_prompts,
        model,
        max_tokens,
        num_return=1,
        best_of=1,
        temperature=0.0,
        top_p=0.0,
        logprobs=None,
        echo=False,
        stop=None,
        timeout=20,
        sleep=1,
        use_async=False,
        num_async_workers=None
):

    # Use async API
    if use_async:
        assert num_async_workers is not None
        # split inputs_with_prompts into num_async_workers batches
        async_inputs_with_prompts = np.array_split(inputs_with_prompts, num_async_workers)
        async_inputs_with_prompts = [list(x) for x in async_inputs_with_prompts]

        completions = asyncio.run(
            async_gpt_inference(
                all_async_prompts=async_inputs_with_prompts,
                model=model,
                max_tokens=max_tokens,
                num_return=num_return,
                best_of=best_of,
                temperature=temperature,
                top_p=top_p,
                logprobs=logprobs,
                echo=echo,
                stop=stop,
                sleep=sleep
            )
        )

    else:
        completions = {"choices": []}
        request_data = {
            "model": model,
            "prompt": inputs_with_prompts,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "n": num_return,
            "logprobs": logprobs,
            "best_of": best_of,
            "echo": echo,
            "stop": stop
        }
        request_headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai.api_key}"
        }
        for i in range(1, 1001):
            try:
                response = requests.post(
                    "https://api.openai.com/v1/completions",
                    json=request_data,
                    headers=request_headers,
                    timeout=timeout
                )
                response.raise_for_status()  # Check for any errors in the response
                completions = response.json()
                break
            except Exception as e:
                print_error(e)
                # if the error code is 400 (bad request), there is something wrong with the request itself
                if e.response is not None and e.response.status_code == 400:
                    print(e.response.json()["error"])
                    raise e
                print(f"tried {i} times, sleep for {sleep} seconds ...")
                time.sleep(sleep)

    outputs = [c["text"] for c in completions["choices"]]
    logprob_tokens = [c['logprobs']['tokens'] for c in completions["choices"] if c['logprobs'] is not None]
    token_logprobs = [c['logprobs']['token_logprobs'] for c in completions["choices"] if c['logprobs'] is not None]
    try:
        top_logprobs = [c['logprobs']['top_logprobs'] for c in completions["choices"] if c['logprobs'] is not None]
    except KeyError:
        top_logprobs = []
    logprob_dict = {
        "tokens": logprob_tokens,
        "token_logprobs": token_logprobs,
        "top_logprobs": top_logprobs
    }
    return outputs, logprob_dict


async def async_chat_inference(
    message_list: List,
    model: str,
    max_tokens: int,
    num_return=1,
    temperature=0.0,
    top_p=0.0,
    stop=None,
    sleep=1
):
    completions = []

    for i in range(1, 301):
        try:
            async_responses = [
                openai.ChatCompletion.acreate(
                    model=model,
                    messages=[message],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    n=num_return,
                    stop=stop
                )
                for message in message_list
            ]
            completions = await asyncio.gather(*async_responses)
            break
        except Exception as e:
            if isinstance(e, openai.InvalidRequestError):
                raise e
            print_error(e)
            if i % 100 == 0:
                print(f"tried {i} times, sleep for 10 minutes ...")
                time.sleep(600)
            else:
                print(f"tried {i} times, sleep for {sleep} seconds ...")
                time.sleep(sleep)

    if not completions:
        raise RuntimeError("No responses collected from OpenAI. "
                           "If this comes from rate limits, please lower the number of async coroutines.")

    # Combine all completion["choices"] into one
    combined_completions = []
    for c in completions:
        combined_completions.extend(c["choices"])
    return {"choices": combined_completions}


def chatgpt_single_turn_inference(
        inputs_with_prompts: List[str],
        model: str,
        max_tokens: int,
        num_return: int = 1,
        temperature: float = 0.0,
        top_p: float = 0.0,
        stop=None,
        timeout: int = 10,
        sleep: int = 1,
        use_async=False,
        num_async_workers=None
):
    # Use async api
    if use_async:
        assert isinstance(inputs_with_prompts, list) and len(inputs_with_prompts) == num_async_workers
        message_list = [{"role": "user", "content": message} for message in inputs_with_prompts]

        completions = asyncio.run(
            async_chat_inference(
                message_list=message_list,
                model=model,
                max_tokens=max_tokens,
                num_return=num_return,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                sleep=sleep
            )
        )
    # Use normal api
    else:
        if isinstance(inputs_with_prompts, list):
            assert len(inputs_with_prompts) == 1, "ChatGPT only allows a single input"
            message = [{"role": "user", "content": inputs_with_prompts[0]}]
        elif isinstance(inputs_with_prompts, str):
            message = [{"role": "user", "content": inputs_with_prompts}]
        else:
            raise ValueError(f"Invalid type for inputs_with_prompts: {type(inputs_with_prompts)}")

        completions = {"choices": []}

        request_data = {
            "model": model,
            "messages": message,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "n": num_return,
            "stop": stop
        }
        request_headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai.api_key}"
        }

        for i in range(1, 1001):
            try:
                response = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    json=request_data,
                    headers=request_headers,
                    timeout=timeout
                )
                response.raise_for_status()  # Check for any errors in the response
                completions = response.json()
                break
            except Exception as e:
                print_error(e)
                # if the error code is 400 (bad request), there is something wrong with the request itself
                if e.response is not None and e.response.status_code == 400:
                    print(e.response.json()["error"])
                    raise e
                print(f"tried {i} times, sleep for {sleep} seconds ...")
                time.sleep(sleep)

    outputs = [c["message"]["content"] for c in completions["choices"]]
    return outputs
