import os
from typing import Optional

import google.generativeai as palm
import openai

from tot.provider import Provider


class OpenAI(Provider):
    def __init__(self, key: Optional[str], model: Optional[str]):
        super().__init__()

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
        response = openai.ChatCompletion.create(
            model=self.model,
            temperature=temperature,
            messages=[{"role": "system", "content": prompt}],
        )

        return response.choices[0].message.content


class PaLM(Provider):
    def __init__(self, key: Optional[str], model: Optional[str]):
        super().__init__()

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


def get_provider(
    provider: Optional[str] = None, model: Optional[str] = None
) -> Provider:
    """
    get_provider returns a provider; if the choice is None, then it will
    check environment variables for the provider to use; openai first,
    then palm. If those aren't set, then it will raise a ValueError.
    """
    if provider is None:
        env_var = os.getenv("OPENAI_API_KEY")
        if env_var is not None and env_var != "":
            provider = "openai"

        env_var = os.getenv("GOOGLE_API_KEY")
        if env_var is not None and env_var != "":
            provider = "palm"

        if provider is None or provider == "":
            raise ValueError("No API key provided")

    if provider == "openai":
        return OpenAI(None, model)
    elif provider == "palm":
        return PaLM(None, model)
    else:
        raise ValueError("Invalid provider")
