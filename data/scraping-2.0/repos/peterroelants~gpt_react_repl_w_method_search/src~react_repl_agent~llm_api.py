import logging
import os
import time
from pathlib import Path
from typing import Any, Final, Literal, Optional

import openai
import openai.error
import requests
import tiktoken

Model = Literal["text-davinci-003", "gpt-3.5-turbo", "gpt-4"]
MODEL_ENV_VAR: Final[str] = "SELECTED_LLM_MODEL"
SEC_IN_MIN: Final[float] = 60
MAX_TOKENS_PER_MIN: Final[int] = 40_000
MAX_REQUEST_PER_MIN: Final[int] = 20
TOKEN_ENCODER: Optional[tiktoken.Encoding] = None


def call_model(
    prompt: str,
    max_tokens: int,
    stop_sequences: list[str],
    nb_retried: float = 0,  # How many times have we retried?
) -> tuple[str, Any]:
    """
    Call the selected LLM model.

    If the model is rate-limited, we wait and retry.
    """
    try:
        return _call_model(
            prompt=prompt,
            max_tokens=max_tokens,
            stop_sequences=stop_sequences,
        )
    except (
        openai.error.RateLimitError,
        openai.error.Timeout,
        openai.error.TryAgain,
        openai.error.APIConnectionError,
        openai.error.ServiceUnavailableError,
        requests.exceptions.Timeout,
    ) as exc:
        logging.warning(f"{type(exc).__name__}: {exc}")
        model = os.environ[MODEL_ENV_VAR]
        logging.info(f"Using '{model}' model!")
        nb_tokens = get_nb_tokens(prompt)
        tpm_wait = (nb_tokens / MAX_TOKENS_PER_MIN) * SEC_IN_MIN
        rpm_wait = (1 / MAX_REQUEST_PER_MIN) * SEC_IN_MIN
        wait = max(tpm_wait, rpm_wait) * (2**nb_retried)  # Exponential back-off
        logging.info(f"Waiting {wait:.2f} seconds before retrying...")
        time.sleep(wait)
        return call_model(
            prompt=prompt,
            max_tokens=max_tokens,
            stop_sequences=stop_sequences,
            nb_retried=nb_retried + 1,
        )


def _call_model(
    prompt: str,
    max_tokens: int,
    stop_sequences: list[str],
) -> tuple[str, Any]:
    model = os.environ[MODEL_ENV_VAR]
    if model == "text-davinci-003":
        return call_text_davinci(prompt, max_tokens, stop_sequences)
    elif model == "gpt-3.5-turbo":
        return call_gpt35turbo(prompt, max_tokens, stop_sequences)
    elif model == "gpt-4":
        return call_gpt4(prompt, max_tokens, stop_sequences)
    else:
        raise ValueError(f"Unknown model: {model}")


def set_model(model: Model):
    global TOKEN_ENCODER
    _set_api_key()
    os.environ[MODEL_ENV_VAR] = model
    TOKEN_ENCODER = tiktoken.encoding_for_model(model)
    logging.info(f"Using '{os.environ[MODEL_ENV_VAR]}' model!")


def call_text_davinci(
    prompt: str,
    max_tokens: int,
    stop_sequences: list[str],
) -> tuple[str, Any]:
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=0.00,
        max_tokens=max_tokens,
        stop=stop_sequences,
    )
    text = response["choices"][0]["text"]
    return text, response


def call_gpt35turbo(
    prompt: str,
    max_tokens: int,
    stop_sequences: list[str],
) -> tuple[str, Any]:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that executes a given task, from start to finish, in steps. All values used in the actions need to come verbatim from the task or observations!",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=max_tokens,
        stop=stop_sequences,
    )
    text = response["choices"][0]["message"]["content"]
    return text, response


def call_gpt4(
    prompt: str,
    max_tokens: int,
    stop_sequences: list[str],
) -> tuple[str, Any]:
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that executes a given task, from start to finish, in steps. All values used in the actions need to come verbatim from the task or observations!",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=max_tokens,
        stop=stop_sequences,
    )
    text = response["choices"][0]["message"]["content"]
    return text, response


def get_nb_tokens(s: str):
    """Estimate the number of tokens in a string."""
    assert TOKEN_ENCODER is not None
    return len(TOKEN_ENCODER.encode(s))


def _set_api_key():
    """Set the OpenAI API key from the environment or a file."""
    OPENAI_API_KEY_NAME = "OPENAI_API_KEY"
    if OPENAI_API_KEY_NAME in os.environ:
        return
    else:
        secret_path = (
            Path(__file__).resolve().parent.parent.parent / "secrets" / "openai_api.key"
        )
        logging.info(f"Reading OpenAI API key from '{secret_path}'.")
        with secret_path.open("r") as f:
            key = f.read().strip()
            os.environ[OPENAI_API_KEY_NAME] = key
            openai.api_key = key
