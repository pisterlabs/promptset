"""Common functions for GPT generation."""
import random
import time
from logging import Logger
from typing import List

import tiktoken
from openai import OpenAI

from config import logger, openai_api_key

# Set the OpenAI model
MODEL = "gpt-4"

client = OpenAI(api_key=openai_api_key)


# TODO Switch to ASync Client
def chat_completion_wrapper(model, messages, temperature: float = 0.7):
    """Wrap the openai chat completion API to allow test substitution."""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature
    )
    response = response.choices[0].message.content
    return response


def api_request(
        text: str = None,
        messages: List[dict] = None,
        model: str = MODEL,
        temperature: float = 0.7,
        gen_logger: Logger = logger
) -> List[List[float]] | str:
    """Make a request to the openai api."""
    max_tries = 5
    initial_delay = 1
    backoff_factor = 2
    max_delay = 16
    jitter_range = (1, 3)

    if model.startswith("text-embedding"):
        # Get rid of newlines
        text = [t.replace("\n", " ") for t in text]

    for attempt in range(1, max_tries + 1):
        try:
            # gen_logger.info(f"Making API request with {model}")
            if model.startswith("text-embedding"):
                gen_logger.info(f"Making API request for text embedding")
                response = client.embeddings.create(input=text, model=model)
                result = response.data[0].embedding
            else:
                gen_logger.info(f"Making API request with {model}")
                result = chat_completion_wrapper(model, messages, temperature=temperature)
            return result
        except Exception as e:
            if attempt == max_tries:
                gen_logger.error(f"API request failed after {attempt} attempts with final error {e}.")
                return []

            delay = min(initial_delay * (backoff_factor ** (attempt - 1)), max_delay)
            jitter = random.uniform(jitter_range[0], jitter_range[1])
            sleep_time = delay + jitter
            gen_logger.error(
                f"API request failed with error: {e}. "
                f"Retrying in {sleep_time:.2f} seconds."
            )
            time.sleep(sleep_time)


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo-0301":  # note: future models may deviate from this
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
            num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.""")



