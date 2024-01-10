import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

from typing import Any

from llmtaskgraph.types import Prompt, JSON


class OpenAiChatApiHandler:
    # todo: support batching
    def __init__(self):
        pass

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    async def api_call(
        self,
        prompt: Prompt,
        params: JSON,
    ) -> str:
        # make sure messages is a list of objects with role and content keys
        if not isinstance(prompt, list):
            if isinstance(prompt, str):
                prompt = {"role": "user", "content": prompt}
            prompt = [prompt]

        # Todo: handle api calls elsewhere for request batching and retries
        response: Any = await openai.ChatCompletion.acreate(  # type: ignore
            messages=prompt,
            **params,
        )
        # Todo: handle n > 1
        return response.choices[0].message.content
