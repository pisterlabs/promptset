# system
import os
import asyncio

# first-party
from sqllite_cache import SQLiteCache

# thrid-party
from dotenv import load_dotenv
import openai

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)


class GPTUtil(object):
    def __init__(self, enable_cache=True):
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")

        self.cache = None
        if enable_cache:
            self.cache = SQLiteCache()

    @retry(wait=wait_random_exponential(min=5, max=60), stop=stop_after_attempt(3))
    def gpt_response(self, user_message, **kwargs):
        messages = [{"role": "user", "content": user_message}]

        if self.cache:
            cached_response = self.cache.get(messages)
            if cached_response:
                return cached_response

        request = dict(
            model=kwargs.get("model", "gpt-3.5-turbo"),
            temperature=kwargs.get("temperature", 0.7),
            messages=messages,
        )

        try:
            response = openai.ChatCompletion.create(**request)
        except openai.InvalidRequestError as e:
            print("Trying with gpt-4...")
            request["model"] = "gpt-4"
            try:
                response = openai.ChatCompletion.create(**request)
            except openai.InvalidRequestError as e:
                print("Invalid request", e)
                return None

        content = response["choices"][0]["message"]["content"]
        if self.cache:
            self.cache.set(messages, content)
        return content

    @retry(wait=wait_random_exponential(min=5, max=60), stop=stop_after_attempt(3))
    async def gpt_response_async(self, user_message, **kwargs):
        messages = [{"role": "user", "content": user_message}]

        if self.cache:
            cached_response = self.cache.get(messages)
            if cached_response:
                return cached_response

        request = dict(
            model=kwargs.get("model", "gpt-3.5-turbo"),
            temperature=kwargs.get("temperature", 0.7),
            messages=messages,
        )

        try:
            response = await openai.ChatCompletion.acreate(**request)
        except openai.InvalidRequestError as e:
            print("Trying with gpt-4...")
            request["model"] = "gpt-4"
            try:
                response = await openai.ChatCompletion.acreate(**request)
            except openai.InvalidRequestError as e:
                print("Invalid request", e)
                return None

        content = response["choices"][0]["message"]["content"]
        if self.cache:
            self.cache.set(messages, content)
        return content

    async def parallel_gpt_response(self, message_list, **kwargs):
        tasks = [self.gpt_response_async(message, **kwargs) for message in message_list]
        responses = await asyncio.gather(*tasks)
        return responses
