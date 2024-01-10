#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

This script can be used to make parallel requests to the OpenAI API.

It is based on the following blog post:
    https://towardsdatascience.com/the-proper-way-to-make-calls-to-chatgpt-api-52e635bea8ff

"""

import asyncio
import aiohttp
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

from helpers import logger
from api_secrets import OPENAI_API_KEY

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {OPENAI_API_KEY}"
}

max_retries = 20

costings = {
        "gpt-3.5-turbo": (0.0015, 0.0020), 
        "gpt-3.5-turbo-16k": (0.003, 0.004), 
        "gpt-4": (0.03, 0.06), # 8k context
        "gpt-3.5-turbo-1106": (0.0010, 0.0020),
    }

def get_cost(model_name, prompt_tokens, completion_tokens):
    prompt_cost = ((prompt_tokens / 1000) * costings[model_name][0])
    completion_cost = ((completion_tokens / 1000) * costings[model_name][1])
    return prompt_cost + completion_cost

class ProgressLog:
    def __init__(self, total, model_name):
        self.total = total
        self.model_name = model_name
        self.done = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cost = 0

    def increment(self, prompt_tokens, completion_tokens, cost):
        self.done = self.done + 1
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.cost += cost

    def __repr__(self):
        return f"{(self.done/self.total)*100}%, completed runs {self.done}/{self.total}, total prompt tokens: {self.prompt_tokens}, total completion tokens: {self.completion_tokens}, total cost: {self.cost:.8f}"

@retry(wait=wait_random_exponential(multiplier=2, min=1, max=60), stop=stop_after_attempt(max_retries), before_sleep=logger.info, retry_error_callback=lambda _: None)
async def get_completion(model_name, messages, session, semaphore, progress_log, seed, temperature, expects_json):
    async with semaphore:

        async with session.post("https://api.openai.com/v1/chat/completions", headers=headers, json={
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "seed": seed,
            "response_format": { "type": "json_object" } if expects_json else None,
        }) as resp:

            response_json = await resp.json()
            
            if response_json.get("error") is not None:
                logger.warning(f"{response_json['error']['type']}: {response_json['error']['message']}")

            cost = get_cost(model_name, response_json["usage"]["prompt_tokens"], response_json["usage"]["completion_tokens"])
            progress_log.increment(response_json["usage"]["prompt_tokens"], response_json["usage"]["completion_tokens"], cost)

            response_dict = {
                "content": response_json["choices"][0]["message"]["content"],
                "system_fingerprint": response_json["system_fingerprint"],
                "prompt_tokens": response_json["usage"]["prompt_tokens"],
                "completion_tokens": response_json["usage"]["completion_tokens"],
                "cost": cost,
                "model_name": model_name,
            }

            # log progress every 10% of the way
            if progress_log.total >= 10:
                if progress_log.done > 0 and progress_log.done % (progress_log.total // 10) == 0:
                    logger.info(progress_log)
            else:
                logger.info(progress_log)
            
            return response_dict

async def get_completion_list(model_name, content_list, max_parallel_calls, timeout, seed, temperature, expects_json):
    semaphore = asyncio.Semaphore(value=max_parallel_calls) # manages the amount of async calls that are currently being performed within its context
    progress_log = ProgressLog(len(content_list), model_name)

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(timeout)) as session:
        return await asyncio.gather(*[get_completion(model_name=model_name, messages=content, session=session, semaphore=semaphore, progress_log=progress_log, seed=seed, temperature=temperature, expects_json=expects_json) for content in content_list])

async def _run_api_request_parallel_process(model_name, content_list, max_parallel_calls, timeout, seed, temperature, expects_json):
    import time
    start_time = time.perf_counter()

    completion_list = await get_completion_list(model_name=model_name, content_list=content_list, max_parallel_calls=max_parallel_calls, timeout=timeout, seed=seed, temperature=temperature, expects_json=expects_json)
    print("Time elapsed: ", time.perf_counter() - start_time, "seconds.")
    print("Total cost: ", sum([completion["cost"] for completion in completion_list if completion is not None]))
    return completion_list
    
# main function to be called from outside
def run_api_request_parallel_process(model_name, content_list, max_parallel_calls=5, timeout=60, seed=42, temperature=0.7, expects_json=False):
    logger.info(f"Running {len(content_list)} parallel requests with max_parallel_calls: {max_parallel_calls}, timeout: {timeout}, seed: {seed}, temperature: {temperature}, expects_json: {expects_json}")
    return asyncio.run(_run_api_request_parallel_process(model_name, content_list, max_parallel_calls=max_parallel_calls, timeout=timeout, seed=seed, temperature=temperature, expects_json=expects_json))

if __name__ == "__main__":

    def prepare_messages(prompts):
        return [[{"role": "user", "content": prompt}] for prompt in prompts]
    prompts = [f"repeat the following number once. {i}" for i in range(5)]
    prompts = prepare_messages(prompts)
    
    # should work
    run_api_request_parallel_process(model_name="gpt-3.5-turbo-1106", content_list=prompts, max_parallel_calls=2, timeout=10, seed=42, temperature=0.0, expects_json=False)

    # should fail with warnings logged
    run_api_request_parallel_process(model_name="gpt-3.5-turbo-1106", content_list=prompts, max_parallel_calls=2, timeout=10, seed=42, temperature=0.0, expects_json=True)
