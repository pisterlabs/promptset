from typing import Optional

import openai
import tiktoken
from loguru import logger
from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from sum_pai.constants import EMBEDDING_ENCODING


@retry(
    wait=wait_random_exponential(min=1, max=20),
    stop=stop_after_attempt(10),
    retry=retry_if_not_exception_type(openai.InvalidRequestError),
)
def chat_completion(content: str, system: Optional[str] = None) -> str:
    messages = [
        {"role": "user", "content": content},
    ]
    if system:
        messages.insert(0, {"role": "system", "content": system})
    response = openai.ChatCompletion.create(
        model=get_model_for_messages(messages),
        messages=messages,
        max_tokens=512,
        n=1,
        stop=None,
        temperature=0.5,
    )
    logger.debug(f"ChatCompletion response: {response}")
    return response.choices[0].message["content"].strip()


def create_tokens(text: str):
    """Creates tokens for the given text using OpenAI's tiktoken library.

    Args:
        text (str): The text to create tokens for.

    Returns:
        list: The list of tokens for the given text.
    """
    encoding = tiktoken.get_encoding(EMBEDDING_ENCODING)
    return encoding.encode(text)


def get_model_for_messages(message: list):
    """Gets the model to use for the given messages.

    Args:
        message (list): The messages to get the model for.

    Returns:
        str: The model to use for the given messages.
    """
    total_tokens = sum(len(create_tokens(message["content"])) for message in message)
    if total_tokens > 4000:
        return "gpt-4"
    if total_tokens > 8000:
        return "gpt-4-32k"
    if total_tokens > 32000:
        logger.critical(
            "Total tokens must be less than 32000, " "your code is too strong!"
        )
        exit(1)
    return "gpt-3.5-turbo"
