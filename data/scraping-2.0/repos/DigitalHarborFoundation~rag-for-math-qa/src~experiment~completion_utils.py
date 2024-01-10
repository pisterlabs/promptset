import logging
import time

import openai

logger = logging.getLogger(__name__)


def get_completion_noraise(
    messages: list,
    sleep: float = 0.1,
    should_log_successful: bool = False,
    **kwargs,
) -> str | None:
    """Function wrapper that swallows exceptions, intended to use with multiprocessing.

    Returns:
        str | None: The completion, or None if an exception was raised.
    """
    try:
        generation = get_completion_with_wait(messages, sleep=sleep, **kwargs)
        if should_log_successful:
            logger.info("Successful completion.")
        return generation
    except Exception as ex:
        logger.warning(f"get_completion_noraise returning None due to {type(ex).__name__} error: {ex}")
        return None


def get_completion_with_wait(messages: list, sleep: float = 0.1, **kwargs) -> str:
    generation = get_completion_with_retries(messages, **kwargs)
    if sleep:
        time.sleep(sleep)  # being a bit polite on repeated api calls
    return generation


def get_completion_with_retries(
    messages: list,
    max_attempts: int = 3,
    sleep_time_between_attempts: float = 5,
    **kwargs,
) -> str:
    """Could use a library for this, but let's keep it simple.

    Args:
        messages (list): _description_
        max_attempts (int, optional): Defaults to 3.
        sleep_time (float, optional): Defaults to 5 (seconds).

    Returns:
        str: The completion
    """
    n_attempts = 0
    while n_attempts < max_attempts:
        n_attempts += 1
        try:
            return get_completion(messages, **kwargs)
        except Exception as ex:
            logger.warning(f"Failure on attempt {n_attempts} / {max_attempts}: {type(ex).__name__} {ex}")
            if n_attempts == max_attempts:
                raise ex
            time.sleep(sleep_time_between_attempts * n_attempts)
    raise ValueError(
        f"Exceeded max attempts ({max_attempts}), base sleep interval {sleep_time_between_attempts}s; this error indicates an unexpected logical flow",
    )


def get_completion(messages: list, model_name: str = "gpt-3.5-turbo-0613", request_timeout: float = 20) -> str:
    completion = openai.ChatCompletion.create(
        model=model_name,
        messages=messages,
        request_timeout=request_timeout,
    )
    assistant_message = completion["choices"][0]["message"]["content"]
    return assistant_message
