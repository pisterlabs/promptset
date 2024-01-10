from functools import partial

import openai
from openai import Completion

from generator import PromptGenerator
from schema.request import Message


class OpenAIService:
    def __init__(self, api_key: str, organization: str, model_name: str, temperature: float) -> None:
        # key settings
        openai.api_key = api_key
        openai.organization = organization

        self._openai_request = partial(
            Completion.create,
            model=model_name,
            temperature=temperature
        )

    def chat(self, message: Message) -> str:
        prompt = PromptGenerator.generate(message.content)
        response = self._openai_request(prompt=prompt)
        return response.choices[0].text
