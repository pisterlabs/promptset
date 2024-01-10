import logging
import os
import time
from typing import Any, Optional

import openai
import openai.error
import tiktoken

openai.api_key = os.getenv("OPENAI_API_KEY")

# Set the organization
openai.organization = os.getenv("OPENAI_ORGANIZATION_ID")

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant tasked with extending a legal dataset. Your sole task is to add"
    " plausible facts to a given fictional court case."
)


class TokenCounter:
    prize_per_token = {
        "gpt-3.5-turbo": 0.002 / 1000,
    }

    def __init__(self, model_name="gpt-3.5-turbo", max_tokens=200):
        self.max_tokens = max_tokens
        self.model_name = model_name

        # Initialize the tokenizer
        self.encoder = tiktoken.encoding_for_model(model_name)

        # Initialize the counters
        self.input_token_counter = 0
        self.num_calls = 0
        self.input_token_sizes = []

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        num_input_tokens = len(self.encoder.encode(prompt))
        self.input_token_counter += num_input_tokens
        self.input_token_sizes.append(num_input_tokens)
        self.num_calls += 1
        return "[DEBUG]"

    @property
    def output_token_counter(self):
        return self.num_calls * self.max_tokens

    @property
    def total_token_counter(self):
        return self.input_token_counter + self.output_token_counter

    @property
    def cost_upper_bound(self):
        return self.total_token_counter * self.prize_per_token[self.model_name]

    def top_k_input_token_lengths(self, k=10):
        return sorted(self.input_token_sizes, reverse=True)[:k]


def chat_message(role: str, content: str):
    return {"role": role, "content": content}


def gpt_query(
    prompt: str,
    system_prompt: Optional[str] = None,
    model_name: str = "gpt-3.5-turbo",
    max_tokens: int = 200,
    temperature: float = 0.8,
    **kwargs: Any,
):
    if system_prompt is None:
        system_prompt = DEFAULT_SYSTEM_PROMPT

    num_seconds_to_wait_max = 300

    if "gpt-3.5" in model_name or "gpt-4" in model_name:
        messages = [
            chat_message("system", system_prompt),
            chat_message("user", prompt),
        ]

        time_waited = 0
        wait_time_seconds = 5
        while time_waited < num_seconds_to_wait_max:
            try:
                completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                )
                break
            except openai.error.APIError as e:
                logging.info(f"APIError: {e}. Waiting {wait_time_seconds} seconds...")
                time.sleep(wait_time_seconds)
                time_waited += wait_time_seconds
            except openai.error.RateLimitError as e:
                logging.info(f"RateLimitError: {e}. Waiting {wait_time_seconds} seconds...")
                time.sleep(wait_time_seconds)
                time_waited += wait_time_seconds
                wait_time_seconds *= 2
        else:
            raise TimeoutError(
                f"Timed out waiting for {model_name} to respond after"
                f" {num_seconds_to_wait_max} seconds."
            )

        return completion.choices[0].message["content"]

    else:
        raise NotImplementedError
