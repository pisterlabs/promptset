from typing import Any, Generic, TypeVar, ParamSpec, Callable

from functools import wraps, lru_cache
import inspect

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

import openai.error


MIN_WAIT_SECONDS = 4
MAX_WAIT_SECONDS = 10
MAX_RETRIES = 5

T = TypeVar("T")
P = ParamSpec("P")

@lru_cache()
def get_function_with_retry(
    func: Callable[P, T],
) -> Callable[P, T]:
    
    @wraps(func)
    def get_decarator() -> Callable[[Callable[P, T]], Callable[P, T]]:
        return retry(
            reraise=True,
            stop=stop_after_attempt(MAX_RETRIES),
            wait=wait_exponential(multiplier=1, min=MIN_WAIT_SECONDS, max=MAX_WAIT_SECONDS),
            retry=(
                retry_if_exception_type(openai.error.Timeout)
                | retry_if_exception_type(openai.error.APIError)
                | retry_if_exception_type(openai.error.APIConnectionError)
                | retry_if_exception_type(openai.error.RateLimitError)
                | retry_if_exception_type(openai.error.ServiceUnavailableError)
            ),
            # before_sleep=before_sleep_log(logger, logging.WARNING),
        )
    
    return get_decarator()(func)



def run_with_retry(
    func: Callable[P, T],
    *args: P.args,
    **kwargs: P.kwargs,
) -> T:
    return get_function_with_retry(func)(*args, **kwargs)


__all__ = ["run_with_retry"]
