# -*- coding: utf-8 -*-
"""Module with client calling the OpenAI GPT completion endpoint"""

from typing import List
from typing import Tuple

import openai
import requests

# ==============================================================================
# CONSTANT DEFINITION
# ==============================================================================

API_EXCEPTIONS = (requests.HTTPError,)

# ==============================================================================
# CLASS AND FUNCTION DEFINITION
# ==============================================================================


class GPTClient:
    def __init__(self, engine, api_key) -> None:
        self.engine = engine
        self.api_key = api_key
        openai.api_key = api_key

    def format_prompt(
        self,
        task: str = "",
        text: str = "",
        input_desc: str = "",
        output_desc: str = "",
        examples: List[Tuple[str, str]] = [("", "")],
    ) -> str:
        """
        Returns prompt of form:

        Correct grammar mistakes.

        Original: Where do you went?
        Standard American English: Where did you go?
        Original: Where is you?
        Standard American English:

        Args:
            task: The task for GPT, e.g. Correct grammar mistakes.
            text: The current row based on which to generate
            input_desc: Description of input column
            output_desc: Description of output column
            example_in: Example of input text
            example_out: Example of desired output text
        Returns:
            prompt: Formatted prompt
        """

        prompt = ""

        ### Task ###
        if task:
            prompt += f"{task}\n\n"

        ### Examples ###
        for ex_inp, ex_out in examples:
            if ex_inp:
                if input_desc:
                    prompt += f"{input_desc}: "
                prompt += f"{ex_inp}\n"

            # One could also provide output examples without descriptions, e.g.
            # elephant
            # giraffe
            # cat
            if ex_out:
                if output_desc:
                    prompt += f"{output_desc}: "
                prompt += f"{ex_out}\n"

        ### Final prompt ###
        if text:
            if input_desc:
                prompt += f"{input_desc}: "
            prompt += f"{text}\n"

        if output_desc:
            # Do not end with a space, as it worsens generation
            prompt += f"{output_desc}:"

        return prompt

    def generate(
        self,
        task: str = "",
        text: str = "",
        input_desc: str = "",
        output_desc: str = "",
        examples: List[Tuple[str, str]] = [("", "")],
        temperature: float = 0.7,
        max_tokens: int = 64,
    ) -> str:
        """
        Constructs a prompt and makes an API call to generate text.
        Default values for temperature and max_tokens are chosen based on the OpenAI playground.
        """
        prompt = self.format_prompt(task, text, input_desc, output_desc, examples)

        response = openai.Completion.create(
            engine=self.engine,
            prompt=prompt,
            stop="\n",
            temperature=temperature,
            max_tokens=max_tokens,
        )

        if "choices" in response:
            return response["choices"][0]
        else:
            # OpenAIs Python client seems to handle all exceptions so this should rarely be called
            user_message = f"Encountered the following error while sending an API request to OpenAI: {response}"
            raise requests.HTTPError(user_message)
