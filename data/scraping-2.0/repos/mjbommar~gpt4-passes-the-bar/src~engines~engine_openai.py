"""
This module implements the OpenAI engine for any text completion models available through the API, including:
 * text-ada-001
 * text-babbage-001
 * text-curie-001
 * text-davinci-001
 * text-davinci-003
"""

# imports
import logging
import os
import time
from itertools import product, combinations_with_replacement
from pathlib import Path
from typing import Iterator

# packages
import numpy.random
import openai

# project import
from engines.base_engine import BaseEngine

# set up logging
logger = logging.getLogger(__name__)

# default parameters
DEFAULT_OPENAI_API_MODEL = "text-davinci-003"

# set default valid parameters
OPENAI_VALID_PARAMETERS = {
    "temperature": [0.0, 0.5, 1.0],
    "max_tokens": [16, 32, 64, 128, 256],
    "best_of": [1, 2, 4],
}


def get_openai_api_key() -> str | None:
    """
    This function returns the OpenAI API key from the environment variable OPENAI_API_KEY.
    Returns:
        str | None: The OpenAI API key or None if it is not set anywhere.
    """
    # check for env variable defined
    if "OPENAI_API_KEY" in os.environ:
        # return the value
        return os.environ["OPENAI_API_KEY"]

    # check for .openai_key file in module directory
    module_key_file = Path(__file__).parent / ".openai_key"
    if module_key_file.exists():
        # return the value
        file_buffer = module_key_file.read_text().strip()
        if file_buffer.count("-") == 1 and 48 <= len(file_buffer) <= 52:
            return file_buffer

    # check for .openai_key file in current directory
    local_key_file = Path("./.openai_key")
    if local_key_file.exists():
        # return the value
        file_buffer = local_key_file.read_text().strip()
        if file_buffer.count("-") == 1 and 48 <= len(file_buffer) <= 52:
            return file_buffer

    # check for .openai_key file in home directory
    home_key_file = Path.home() / ".openai_key"
    if home_key_file.exists():
        # return the value
        file_buffer = home_key_file.read_text().strip()
        if file_buffer.count("-") == 1 and 48 <= len(file_buffer) <= 52:
            return file_buffer

    # if we get here, we didn't find a key
    return None


class OpenAIEngine(BaseEngine):
    """
    OpenAI text completion engine implementing base engine interface.
    """

    def __init__(
        self,
        model: str = DEFAULT_OPENAI_API_MODEL,
        api_key: str | None = None,
        parameters: dict | None = None,
        sleep_time: float = 0.0,
        retry_count: int = 3,
        retry_time: float = 5.0,
        seed: int | None = None,
    ) -> None:
        """
        Constructor for the engine.
        :param model:
        :param parameters:
        """

        # set the api key
        if api_key is None:
            api_key = get_openai_api_key()
        if api_key is None:
            raise ValueError("No API key found for OpenAI API.")
        self.api_key = openai.api_key = api_key

        # set the model
        self.model = model

        # set the parameters
        if parameters is None:
            parameters = {}
        self.parameters = parameters

        # handle retry and sleep settings
        self.sleep_time = sleep_time
        self.retry_count = retry_count
        self.retry_time = retry_time

        # setup rng
        if seed is None:
            seed = numpy.random.randint(0, 2**32 - 1, dtype=numpy.int64)
        self.rng = numpy.random.RandomState(seed)

    def get_name(self) -> str:
        """
        This method returns the name of the engine.
        Returns:
            str: The name of the engine with namespace:model format.
        """
        return f"openai:{self.model}"

    def get_current_parameters(self) -> dict:
        """
        This method returns the current parameters of the engine.
        Returns:
            dict: The current parameters of the engine.
        """
        return self.parameters

    def get_valid_parameters(self) -> Iterator[dict]:
        """
        This method returns a set of valid parameters.
        Returns:
            Iterator[dict]: An iterator of valid parameters.
        """
        # get all combinations of values from DEFAULT_OPENAI_VALID_PARAMETERS with itertools/functools
        for parameter_combination in product(*OPENAI_VALID_PARAMETERS.values()):
            # convert to dict
            parameter_dict = dict(
                zip(OPENAI_VALID_PARAMETERS.keys(), parameter_combination)
            )
            # yield the parameter dict
            yield parameter_dict

    def get_random_parameters(self) -> dict:
        """
        This method returns a random set of parameters.
        Returns:
            dict: A random set of parameters.
        """
        # get a random combination of values from DEFAULT_OPENAI_VALID_PARAMETERS with itertools/functools
        parameter_combination = self.rng.choice(
            list(
                combinations_with_replacement(
                    OPENAI_VALID_PARAMETERS.values(), len(OPENAI_VALID_PARAMETERS)
                )
            )
        )

        # convert to dict
        parameter_dict = dict(
            zip(OPENAI_VALID_PARAMETERS.keys(), parameter_combination)
        )

        # return the parameter dict
        return parameter_dict

    def set_parameters(self, parameters: dict) -> None:
        """
        This method sets the parameters of the engine.
        N.B.: This does NOT update parameters.  To do so, create a copy with `get_current_parameters` and
        set from the updated copy.
        Args:
            parameters (dict): The parameters to set.
        """
        # set the parameters
        self.parameters = parameters

    def get_completions(self,
                        prompt: str,
                        n: int = 1,
                        system_prompt: str | None = None,
                        parameters: dict | None = None
                        ) -> list[str]:
        """
        This method returns n completions for the prompt.
        Args:
            prompt (str): The prompt to complete.
            n (int): The number of completions to return.
            system_prompt (str): The system prompt to use for ChatCompletion APIs.
            parameters (dict): The parameters to use for the completion.

        Returns:
            list[dict]: A list of completion dictionaries.
        """
        # initial sleep
        time.sleep(self.sleep_time)

        # update with any parameters
        if parameters is not None:
            self.parameters.update(parameters)

        # response list
        response_list = []

        # setup retry tracker and get requested number of completions
        retry_count = 0

        for _ in range(n):
            response = None
            while response is None and retry_count < self.retry_count:
                if retry_count > self.retry_count:
                    raise RuntimeError("Too many retries.")

                try:
                    if system_prompt is not None:
                        # we're in a chat context, so use the chat API
                        messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt},
                        ]

                        response = openai.ChatCompletion.create(
                            model=self.model,
                            messages=messages,
                            **self.parameters
                        )

                        # parse chat field structure
                        if len(response["choices"]) > 0:
                            response_list.append(response["choices"][-1]["message"]["content"])
                        else:
                            response = None
                    else:
                        response = openai.Completion.create(
                            model=self.model, prompt=prompt, **self.parameters
                        )

                        # parse completion field structure
                        if len(response["choices"]) > 0:
                            response_list.append(response["choices"][0]["text"])
                        else:
                            response = None
                except Exception as e:
                    logger.error("OpenAI API error (try=%d): %s", retry_count, e)
                    retry_count += 1

                    # sleep
                    time.sleep(self.retry_time)
                    continue

        # return the completions
        return response_list
