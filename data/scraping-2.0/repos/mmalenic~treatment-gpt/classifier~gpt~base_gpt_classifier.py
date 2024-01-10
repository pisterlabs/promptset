import asyncio
import json
import os
from abc import ABC, abstractmethod
from decimal import *
from json import JSONDecoder
from pathlib import Path
from typing import Literal, List, Any

import numpy as np
import openai
import pandas as pd
import tiktoken
from openai.error import Timeout, ServiceUnavailableError, APIError, RateLimitError

from classifier.gpt.prompt_templates import Prompts
from dataset.utils import process_plus


class BaseGPTClassifier(ABC):
    output_n_tokens_estimate = 200
    output_n_tokens_estimate_cot = 1000
    max_4k_input = 4097 - output_n_tokens_estimate
    max_4k_input_cot = 4097 - output_n_tokens_estimate_cot
    max_8k_input = 8192 - output_n_tokens_estimate
    max_8k_input_cot = 8192 - output_n_tokens_estimate_cot

    def __init__(
        self,
        df: pd.DataFrame,
        base_dataset,
        save_dir: str,
        model_type: Literal["gpt-3.5-turbo"]
        | Literal["gpt-3.5-turbo-16k"]
        | Literal["gpt-4"]
        | Literal["gpt-4-32k"] = "gpt-3.5-turbo",
        repeat_n_times: int = 1,
        batch_n: int = None,
        **kwargs,
    ):
        """
        Initialize this class.
        """
        super().__init__()
        self.__dict__.update(kwargs)

        self._model_type = model_type
        self.base_dataset = base_dataset
        self.df = df
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        self.save_dir = save_dir
        self._cost_estimate = None
        if batch_n is None:
            self._batch_n = (
                3 if model_type == "gpt-4" or model_type == "gpt-4-32k" else 10
            )
        self._max_token_number = None
        self._repeat_n_times = repeat_n_times

    def n_samples(self) -> int:
        return len(self.df)

    @property
    def cost_estimate(self):
        """
        Estimate costs
        """

        def apply_estimates(x):
            prompt = self._construct_prompt(x)
            n_tokens = self._n_tokens(prompt)
            model_type = self._get_model_type(n_tokens, prompt)
            estimate = self._cost_estimate_single(model_type, n_tokens)

            if self._max_token_number is None or n_tokens > self._max_token_number:
                self._max_token_number = n_tokens

            estimates.append(estimate * self._repeat_n_times)

        estimates = []
        self.df.apply(apply_estimates, axis=1)

        self._cost_estimate = sum(estimates)

        return self._cost_estimate

    @property
    def max_token_number(self):
        """
        Maximum number of tokens in the dataset
        """
        return self._max_token_number

    def save_example_prompt(self):
        """
        Save an example prompt.
        """
        path = Path(os.path.join(self.save_dir, "example_prompt"))
        path.write_text(self._construct_prompt(self.df.iloc[0]), encoding="utf-8")

    async def predict(self):
        """
        Predict the labels.
        """
        print("batching predict values:", self._batch_n)
        dfs = []
        for x in np.split(
            self.df, np.arange(self._batch_n, len(self.df), self._batch_n)
        ):
            batch = await self._predict_batch(x)
            x = pd.concat(batch, axis=1).reset_index().transpose()
            x = x.rename(columns=x.iloc[0]).drop(x.index[0])
            dfs.append(x)

        self.df = pd.concat(dfs)
        self.df = self.df.explode(column="y_pred", ignore_index=True)
        self.df = self.df.apply(lambda x: self._results(x), axis=1)

        self.base_dataset.df = self.df

    def _n_tokens(self, prompt) -> int:
        """
        Get the number of tokens in a prompt.
        """
        encoding = tiktoken.encoding_for_model(self._model_type)
        return len(encoding.encode(prompt))

    def _cost_estimate_single(self, model_type: str, n_tokens: int) -> Decimal:
        """
        Cost for a single sample.
        """
        if model_type == "gpt-3.5-turbo":
            return ((Decimal(n_tokens) / Decimal(1000)) * Decimal(0.0015)) + (
                (Decimal(self.output_n_tokens_estimate) / Decimal(1000))
                * Decimal(0.002)
            )
        elif model_type == "gpt-3.5-turbo-16k":
            return ((Decimal(n_tokens) / Decimal(1000)) * Decimal(0.003)) + (
                (Decimal(self.output_n_tokens_estimate) / Decimal(1000))
                * Decimal(0.004)
            )
        elif model_type == "gpt-4":
            return ((Decimal(n_tokens) / Decimal(1000)) * Decimal(0.03)) + (
                (Decimal(self.output_n_tokens_estimate) / Decimal(1000)) * Decimal(0.06)
            )
        else:
            return ((Decimal(n_tokens) / Decimal(1000)) * Decimal(0.06)) + (
                (Decimal(self.output_n_tokens_estimate) / Decimal(1000)) * Decimal(0.12)
            )

    def _get_model_type(self, n_tokens: int, prompt: str) -> str:
        """
        Get the best model type based on the number of tokens.
        """
        model_type = self._model_type

        if (
            Prompts.cot_prompt_template in prompt
            or Prompts.treatment_only_cot_prompt_template in prompt
        ):
            if model_type == "gpt-3.5-turbo" and n_tokens > self.max_4k_input_cot:
                model_type = "gpt-3.5-turbo-16k"
            if model_type == "gpt-4" and n_tokens > self.max_8k_input_cot:
                model_type = "gpt-4-32k"

        if (
            Prompts.cot_prompt_template not in prompt
            and Prompts.treatment_only_cot_prompt_template not in prompt
        ):
            if model_type == "gpt-3.5-turbo" and n_tokens > self.max_4k_input:
                model_type = "gpt-3.5-turbo-16k"
            if model_type == "gpt-4" and n_tokens > self.max_8k_input:
                model_type = "gpt-4-32k"

        return model_type

    async def _predict_batch(self, x, max_retries: int = 3) -> List[pd.DataFrame]:
        """
        Predict some samples.
        """
        tasks = []
        for index, row in x.iterrows():
            tasks.append(self._predict_single(row, max_retries))

        out = await asyncio.gather(*tasks)
        return out

    async def _predict_single(
        self, x, max_retries: int = 3, max_rate_limit_retries: int = 50
    ) -> pd.DataFrame:
        """
        Predict a single sample.
        """

        def dump_response(file, _x, _response, _prompt, unique=False):
            if unique:
                counter = 1
                process_file = file
                while os.path.exists(process_file):
                    process_file = f"{file}_{counter}"
                    counter += 1

            with open(file, "w", encoding="utf-8") as location:
                json.dump(
                    {"x": _x.to_dict(), "response": _response, "prompt": _prompt},
                    location,
                    indent=2,
                )

        index = self._index(x)

        load_from = os.path.join(self.save_dir, index)
        if Path(load_from).exists():
            print("Loading from file:", load_from)
            with open(os.path.join(self.save_dir, index), "r", encoding="utf-8") as f:
                response = json.load(f)["response"]
        else:
            prompt = self._construct_prompt(x)
            print("prompt:", prompt)

            n_tokens = self._n_tokens(prompt)
            model_type = self._get_model_type(n_tokens, prompt)

            try:
                response = await openai.ChatCompletion.acreate(
                    model=model_type,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a text classification model.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    n=self._repeat_n_times,
                    timeout=60,
                    request_timeout=60,
                )
            except (Timeout, ServiceUnavailableError, APIError) as e:
                if max_retries == 0:
                    raise e

                print("Service unavailable, retrying in 10 seconds.")
                await asyncio.sleep(10)
                return await self._predict_single(
                    x, max_retries - 1, max_rate_limit_retries
                )
            except RateLimitError as e:
                if max_rate_limit_retries == 0:
                    raise e

                print("Rate limit reached, waiting 60 seconds to try again.")
                await asyncio.sleep(60)
                return await self._predict_single(
                    x, max_retries, max_rate_limit_retries - 1
                )

            for choice in response["choices"]:
                if choice["finish_reason"] == "length":
                    raise ValueError("Model returned a truncated response.")

        responses = []
        for choice in response["choices"]:
            content = choice["message"]["content"]

            try:
                responses.append(self._extract_response(content))
            except ValueError as e:
                print("Skipping this response:", e)
                dump_response(
                    os.path.join(self.save_dir, f"{index}_error"),
                    x,
                    response,
                    self._construct_prompt(x),
                    unique=True,
                )

        print("responses:", responses)

        try:
            responses = [
                [process_plus(y, self.base_dataset) for y in x]
                if isinstance(x, list)
                else process_plus(x, self.base_dataset)
                for x in responses
            ]
        except AttributeError as e:
            print("Skipping this response:", e)
            dump_response(
                os.path.join(self.save_dir, f"{index}_error"),
                x,
                response,
                self._construct_prompt(x),
                unique=True,
            )

        x["y_pred"] = responses

        if not Path(os.path.join(self.save_dir, index)).exists():
            dump_response(
                os.path.join(self.save_dir, index),
                x,
                response,
                self._construct_prompt(x),
            )

        return x

    def _extract_response(self, content: str):
        """
        Extract the response.
        """

        def decode_dict(json_dict):
            try:
                results.append(json_dict["treatment"])
            except KeyError:
                pass
            try:
                results.append(json_dict["treatments"])
            except KeyError:
                pass
            return json_dict

        json_content = self._extract_json_objects(content)
        if len(json_content) > 1:
            raise ValueError("More than one JSON object found.")
        if len(json_content) == 0:
            raise ValueError("No JSON object found.")
        json_content = json_content[0]

        results = []
        json.loads(json.dumps(json_content), object_hook=decode_dict)

        if len(results) > 1:
            raise ValueError("More than one label found.")
        if len(results) == 0:
            raise ValueError("No label found.")

        return results[0]

    @staticmethod
    def _extract_json_objects(text, decoder=JSONDecoder()) -> List[Any]:
        """
        Find JSON objects in text, and yield the decoded JSON data
        """

        pos = 0
        results = []
        while True:
            match = text.find("{", pos)
            if match == -1:
                break
            try:
                result, index = decoder.raw_decode(text[match:])
                results.append(result)
                pos = match + index
            except ValueError:
                pos = match + 1

        return results

    @abstractmethod
    def _construct_prompt(self, x) -> str:
        raise NotImplementedError

    @abstractmethod
    def _index(self, x) -> str:
        raise NotImplementedError

    @abstractmethod
    def _results(self, x) -> pd.DataFrame:
        raise NotImplementedError
