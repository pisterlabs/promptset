from abc import ABC, abstractmethod
from typing import Any
from os import path
import json

import openai
from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
)

from doomer.discord_utils import hundo_to_float
from doomer.settings import SETTINGS_DIR


class LanguageModel(ABC):
    def __init__(self, model_name: str, settings: dict) -> None:
        self.model_name = model_name
        self.settings = settings
        if path.exists(SETTINGS_DIR / f"{model_name}.json"):
            with open(SETTINGS_DIR / f"{model_name}.json", "r") as infile:
                self.settings.update(json.load(infile))

    @abstractmethod
    def completion_handler(self, prompt: str, max_tokens: int, stop: list):
        raise NotImplementedError

    @abstractmethod
    def parse_completion(self, completion: Any) -> str:
        raise NotImplementedError


class GPT3LanguageModel(LanguageModel):
    def __init__(self, model_name: str) -> None:
        settings = {
            "temperature": 100,
            "frequency_penalty": 0,
            "presence_penalty": 50,
        }
        super().__init__(model_name, settings)

    def completion_handler(self, prompt: str, max_tokens: int, stop: list = None):
        return openai.Completion.create(
            engine="davinci",
            prompt=prompt,
            max_tokens=max_tokens,
            frequency_penalty=hundo_to_float(self.settings["frequency_penalty"]),
            temperature=hundo_to_float(self.settings["temperature"]),
            presence_penalty=hundo_to_float(self.settings["presence_penalty"]),
            stop=stop,
        )

    def parse_completion(self, completion: Any) -> str:
        return completion.choices[0].text


class GPT2TransformersLanguageModel(LanguageModel):
    def __init__(self, tokenizer_name: str, model_name: str, stop: list = None) -> None:
        self._tokenizer = self.update_tokenizer(tokenizer_name)
        self._model = self.update_model(model_name)
        settings = {"temperature": 100, "top_p": 100, "top_k": 0, "max_length": 1024}
        super().__init__(model_name, settings)

    def update_tokenizer(self, tokenizer_name: str):
        return GPT2TokenizerFast.from_pretrained(tokenizer_name)

    def update_model(self, model_name: str):
        return GPT2LMHeadModel.from_pretrained(model_name)

    def completion_handler(
        self, prompt: str, max_tokens: int = None, stop: list = None
    ):
        if not max_tokens:
            max_tokens = self.max_length

        inputs = self._tokenizer(prompt, return_tensors="pt")
        input_len = len(inputs["input_ids"][0])
        full_completion = self._model.generate(
            **inputs,
            do_sample=True,
            max_length=max_tokens,
            top_p=hundo_to_float(self.settings["top_p"]),
            top_k=hundo_to_float(self.settings["top_k"]),
        )
        completion = full_completion[0][input_len:]
        completion.resize_(1, len(completion))
        return completion

    def parse_completion(self, completion: Any) -> str:
        return self._tokenizer.decode(completion[0], skip_special_tokens=True)
