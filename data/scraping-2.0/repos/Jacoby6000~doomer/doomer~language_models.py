from abc import ABC, abstractmethod
from typing import Any
from os import path
import json

import openai
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import requests

from doomer.discord_utils import curlify, hundo_to_float
from doomer.settings import SETTINGS_DIR


class LanguageModel(ABC):
    def __init__(self, model_name: str, settings: dict) -> None:
        self.model_name = model_name
        self.settings = settings
        if path.exists(SETTINGS_DIR / f"{model_name}.json"):
            with open(SETTINGS_DIR / f"{model_name}.json", "r") as infile:
                self.settings.update(json.load(infile))

    @abstractmethod
    def completion_handler(self, prompt: str, **kwargs: Any) -> str:
        raise NotImplementedError


class OpenAIGPT3LanguageModel(LanguageModel):
    def __init__(self, model_name: str = "openai-gpt3") -> None:
        settings = {
            "temperature": 100,
            "frequency_penalty": 0,
            "presence_penalty": 50,
        }
        super().__init__(model_name, settings)

    def completion_handler(self, prompt: str, max_tokens: int, stop: list = None):
        completion = openai.Completion.create(
            engine="davinci",
            prompt=prompt,
            max_tokens=max_tokens,
            frequency_penalty=hundo_to_float(self.settings["frequency_penalty"]),
            temperature=hundo_to_float(self.settings["temperature"]),
            presence_penalty=hundo_to_float(self.settings["presence_penalty"]),
            stop=stop,
        )
        completion_text = completion.choices[0].text
        return completion_text


class ExafunctionGPTJLanguageModel(LanguageModel):
    def __init__(self, api_key: str, model_name: str = "exafunction-gptj") -> None:
        settings = {"temperature": 100, "min_tokens": 0}
        self.api_url = "https://nlp-server.exafunction.com/text_completion"
        self.api_key = api_key
        super().__init__(model_name, settings)

    def completion_handler(self, prompt: str, max_tokens: int, **kwargs: any):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "prompt": prompt,
            "max_length": max_tokens,
            "min_length": self.settings["min_tokens"],
            "temperature": hundo_to_float(self.settings["temperature"]),
            "remove_input": "true",
        }
        response = requests.post(self.api_url, json=payload, headers=headers)
        completion = response.json()
        completion_text = completion["text"]
        return completion_text


class AI21JurassicLanguageModel(LanguageModel):
    def __init__(
        self,
        api_key: str,
        model_type: str = "j1-jumbo",
        model_name: str = "ai21-jurassic",
    ) -> None:
        settings = {
            "model_type": model_type,
            "temperature": 100,
            "top_p": 100,
            "max_tokens": 16,
        }
        self.api_key = api_key
        super().__init__(model_name, settings)

    @property
    def api_url(self) -> str:
        return f"https://api.ai21.com/studio/v1/{self.settings['model_type']}/complete"

    def completion_handler(
        self, prompt: str, max_tokens: int, stop: list = None, **kwargs: any
    ):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "prompt": prompt,
            "maxTokens": max_tokens,
            "temperature": hundo_to_float(self.settings["temperature"]),
            "topP": hundo_to_float(self.settings["top_p"]),
            "stopSequences": stop if stop else [],
        }
        response = requests.post(self.api_url, json=payload, headers=headers)
        completion = response.json()
        completion_text = completion["completions"][0]["data"]["text"]
        return completion_text


class GPT2TransformersLanguageModel(LanguageModel):
    def __init__(self, tokenizer_name: str, model_name: str) -> None:
        self.tokenizer_name = tokenizer_name
        self.model_name = model_name
        self._tokenizer = None
        self._model = None
        settings = {"temperature": 100, "top_p": 100, "top_k": 0, "max_length": 1024}
        super().__init__(model_name, settings)

    def update_tokenizer(self, tokenizer_name: str):
        return GPT2TokenizerFast.from_pretrained(tokenizer_name)

    def update_model(self, model_name: str):
        return GPT2LMHeadModel.from_pretrained(model_name)

    def initialize(self):
        self._tokenizer = self.update_tokenizer(self.tokenizer_name)
        self._model = self.update_model(self.model_name)

    def completion_handler(self, prompt: str, max_tokens: int = None, **kwargs: Any):
        if not self._tokenizer or not self._model:
            self.initialize()

        if not max_tokens:
            max_tokens = self.max_length

        inputs = self._tokenizer(prompt, return_tensors="pt")
        input_len = len(inputs["input_ids"][0])
        full_completion = self._model.generate(
            **inputs,
            do_sample=True,
            max_length=input_len + max_tokens,
            top_p=hundo_to_float(self.settings["top_p"]),
            top_k=hundo_to_float(self.settings["top_k"]),
        )

        completion = full_completion[0][input_len:]
        completion.resize_(1, len(completion))
        completion_text = self._tokenizer.decode(
            completion[0], skip_special_tokens=True
        )
        return completion_text
