from typing import Any, Dict, Generator, List

import openai

from g3.config import openai_config as config


class OpenAI:
    def __init__(self):
        openai.api_key = config.api_key
        if config.api_type:
            openai.api_type = config.api_type
        if config.api_base:
            openai.api_base = config.api_base
        if config.api_version:
            openai.api_version = config.api_version

    def generate(self, prompt: List[Dict] | str, **kwargs: Any) -> str:
        kwargs = config.args | kwargs
        resp = openai.Completion.create(
            prompt=prompt,
            **kwargs,
        )
        return resp["choices"][0]["text"]


class OpenAIChat(OpenAI):
    def stream(self, prompt: List[Dict] | str, **kwargs: Any) -> Generator:
        kwargs = config.args | kwargs
        resp = openai.ChatCompletion.create(
            messages=prompt,
            stream=True,
            **kwargs,
        )

        for chunk in resp:
            if "content" in chunk["choices"][0]["delta"]:
                yield chunk["choices"][0]["delta"]["content"]
