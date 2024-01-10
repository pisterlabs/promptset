import os
import re
from functools import partial
from re import Pattern
from typing import Any, Mapping, Optional, Union
import datetime
import time

try:
    from langchain.callbacks.manager import CallbackManagerForLLMRun
    from langchain.llms.base import LLM
except ModuleNotFoundError:
    raise ModuleNotFoundError("Please install langchain: pip install langchain")
from pydantic import Field

from paradigm_client.communicator import SagemakerCommunicator
from paradigm_client.remote_model import RemoteModel
from paradigm_client.request import CreateParameters

DEFAULT_BASE_ADDRESS = "https://paradigm.lighton.ai"


class ParadigmLLM(LLM):
    def __init__(self, client, **kwargs):
        super().__init__(client=client, **kwargs)

    client: RemoteModel
    n_tokens: int = 20  # number of tokens to generate
    temperature: float = 0.7  # temperature to apply to the logits
    top_p: float = 0.9  # p parameter for nucleus sampling
    n_completions: int = 1  # number of generated samples per input
    generate_stop: bool = True
    seed: Optional[int] = None  # set the seed for the sampling phase
    show_special_tokens: bool = False
    biases: dict[int, float] = Field(default_factory=dict)
    stop_sequences: Optional[list[str]] = ["<end_message>"]
    prettify: bool = True
    return_log_probs: bool = False
    echo: bool = False

    def _default_params(self) -> dict[str, Any]:
        return {
            "n_tokens": self.n_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "n_completions": self.n_completions,
            "generate_stop": self.generate_stop,
            "seed": self.seed,
            "show_special_tokens": self.show_special_tokens,
            "biases": self.biases,
            "stop_sequences": self.stop_sequences,
            "prettify": self.prettify,
            "return_log_probs": self.return_log_probs,
            "echo": self.echo,
        }

    @property
    def _llm_type(self) -> str:
        return "paradigm"

    def _current_prefix(self):
        date = datetime.datetime.fromtimestamp(time.time())
        prefix = "You are Alfred, a helpful assistant trained by LightOn. Knowledge cutoff: November 2022. Current date: {date}"
        return prefix.format(date=date.strftime("%B %d, %Y"))

    def _wrap_prompt_to_chatml(self, prompt: str):
        return (
            "<start_system>"
            + self._current_prefix()
            + "<end_message>"
            + "<start_user>"
            + prompt
            + "<end_message><start_assistant>"
        )

    def _call(
        self,
        prompt: str,
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        params = CreateParameters(
            **self._default_params(),
        )
        if stop:
            params.stop_sequences = stop
        else:
            params.stop_sequences = self.stop_sequences
        if "<end_message>" not in params.stop_sequences:
            params.stop_sequences.append("<end_message>")
        text_callback = None
        if run_manager:
            text_callback = partial(run_manager.on_llm_new_token, verbose=self.verbose)
        text = ""
        prompt = self._wrap_prompt_to_chatml(prompt)
        try:
            for completion in self.client.create(
                prompt=prompt,
                params=params,
            ).completions:
                if text_callback:
                    text_callback(completion)
                text += completion.output_text
            return text
        except Exception as e:
            raise e

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return self._default_params()

    def get_num_tokens(self, text: str) -> int:
        return self.client.tokenize(text).n_tokens
