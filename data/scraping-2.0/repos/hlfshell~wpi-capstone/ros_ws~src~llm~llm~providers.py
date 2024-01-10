import os
from abc import ABC, abstractmethod
from typing import Optional

import google.generativeai as palm
import openai


class Provider(ABC):
    @abstractmethod
    def prompt(self, prompt: str, temperature: float) -> str:
        pass


class OpenAI(Provider):
    def __init__(self, key: Optional[str], model: Optional[str]):
        if key is None:
            env_var = os.getenv("OPENAI_API_KEY")
            if env_var is None or env_var == "":
                raise ValueError("No API key provided")

            openai.api_key = env_var
        else:
            openai.api_key = key

        if model is None or model == "":
            self.model = "gpt-3.5-turbo-instruct"
        else:
            self.model = model

    def prompt(self, prompt: str, temperature: float) -> str:
        response = openai.Completion.create(
            model=self.model, temperature=temperature, prompt=prompt
        )
        return response.choices[0].text


class PaLM(Provider):
    def __init__(self, key: Optional[str], model: Optional[str]):
        if key is None:
            env_var = os.getenv("GOOGLE_API_KEY")
            if env_var is None or env_var == "":
                raise ValueError("No API key provided")

            api_key = env_var
        else:
            api_key = key
        palm.configure(api_key=api_key)

        if model is None or model == "":
            self.model = "models/text-bison-001"
        else:
            self.model = model

    def prompt(self, prompt: str, temperature: float) -> str:
        completion = palm.generate_text(
            model=self.model,
            prompt=prompt,
            temperature=temperature,
        )

        return completion.result
