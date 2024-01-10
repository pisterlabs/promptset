# The MIT License (MIT)
# Copyright © 2021 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
import os
import time
import openai
import argparse
import bittensor
import openminers
from typing import List, Dict, Optional


class OpenAIMiner(openminers.BasePromptingMiner):
    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        parser.add_argument(
            "--openai.suffix",
            type=str,
            default=None,
            help="The suffix that comes after a completion of inserted text.",
        )
        parser.add_argument(
            "--openai.max_tokens",
            type=int,
            default=100,
            help="The maximum number of tokens to generate in the completion.",
        )
        parser.add_argument(
            "--openai.temperature",
            type=float,
            default=0.4,
            help="Sampling temperature to use, between 0 and 2.",
        )
        parser.add_argument(
            "--openai.top_p",
            type=float,
            default=1,
            help="Nucleus sampling parameter, top_p probability mass.",
        )
        parser.add_argument(
            "--openai.n",
            type=int,
            default=1,
            help="How many completions to generate for each prompt.",
        )
        parser.add_argument(
            "--openai.presence_penalty",
            type=float,
            default=0.1,
            help="Penalty for tokens based on their presence in the text so far.",
        )
        parser.add_argument(
            "--openai.frequency_penalty",
            type=float,
            default=0.1,
            help="Penalty for tokens based on their frequency in the text so far.",
        )
        parser.add_argument(
            "--openai.model_name",
            type=str,
            default="gpt-3.5-turbo",
            help="OpenAI model to use for completion.",
        )

    @classmethod
    def config(cls) -> "bittensor.Config":
        parser = argparse.ArgumentParser(description="OpenAI Miner Configs")
        cls.add_args(parser)
        return bittensor.config(parser)

    def __init__(self, api_key: Optional[str] = None, *args, **kwargs):
        super(OpenAIMiner, self).__init__(*args, **kwargs)
        if api_key is None:
            raise ValueError(
                "OpenAI API key is None: the miner requires an `OPENAI_API_KEY` defined in the environment variables or as an direct argument into the constructor."
            )
        if self.config.wandb.on:
            self.wandb_run.tags = self.wandb_run.tags + ("openai_miner",)
        openai.api_key = api_key

    def forward(self, messages: List[Dict[str, str]]) -> str:
        resp = openai.ChatCompletion.create(
            model=self.config.openai.model_name,
            messages=messages,
            temperature=self.config.openai.temperature,
            max_tokens=self.config.openai.max_tokens,
            top_p=self.config.openai.top_p,
            frequency_penalty=self.config.openai.frequency_penalty,
            presence_penalty=self.config.openai.presence_penalty,
            n=self.config.openai.n,
        )["choices"][0]["message"]["content"]
        return resp


if __name__ == "__main__":
    openai_api_key = os.getenv("OPENAI_API_KEY")

    with OpenAIMiner(api_key=openai_api_key):
        while True:
            print("running...", time.time())
            time.sleep(1)
