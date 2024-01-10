import email.utils
import time
import typing
from functools import wraps
from random import random
from time import sleep

from loguru import logger

F = typing.TypeVar("F", bound=typing.Callable)
R = typing.TypeVar("R")

MAX_RETRIES = 10


def openai_should_retry(e: Exception) -> bool:
    """
    https://platform.openai.com/docs/guides/error-codes
    """
    import openai

    if isinstance(e, openai.APIStatusError) and e.response.headers:
        # If the server explicitly says whether or not to retry, obey.
        should_retry_header = e.response.headers.get("x-should-retry")
        if should_retry_header == "true":
            return True
        if should_retry_header == "false":
            return False

    good_exc = (
        openai.APIConnectionError,
        openai.APITimeoutError,
        openai.APIStatusError,
    )
    bad_exc = (
        openai.BadRequestError,
        openai.AuthenticationError,
        openai.PermissionDeniedError,
        openai.NotFoundError,
    )
    return isinstance(e, good_exc) and not isinstance(e, bad_exc)


def vertex_ai_should_retry(e: Exception) -> bool:
    import google.api_core.exceptions

    return isinstance(
        e,
        (
            google.api_core.exceptions.ServiceUnavailable,
            google.api_core.exceptions.TooManyRequests,
            google.api_core.exceptions.InternalServerError,
            google.api_core.exceptions.GatewayTimeout,
        ),
    )


def try_all(*fns: typing.Callable[[], R]) -> R:
    assert len(fns) > 0, "Must provide at least one fn"
    prev_exc = None
    for i, fn in enumerate(fns):
        if prev_exc:
            logger.warning(f"[{i + 1}/{len(fns)}] tyring next fn, {prev_exc=}")
        try:
            return fn()
        except Exception as e:
            set_root_cause(e, prev_exc)
            prev_exc = e
    raise prev_exc


def retry_if(
    shuld_retry_fn: typing.Callable[[Exception], bool],
    *,
    max_retries: int = MAX_RETRIES,
    initial_retry_delay: float = 0.5,
    max_retry_delay: float = 8.0,
) -> typing.Callable[[F], F]:
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            prev_exc = None
            assert max_retries, "max_retries must be > 0"
            for idx in range(max_retries + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:
                    set_root_cause(exc, prev_exc)
                    prev_exc = exc
                    if not shuld_retry_fn(exc):
                        break
                    retry_delay = calculate_retry_delay(
                        exc=exc,
                        idx=idx,
                        initial_retry_delay=initial_retry_delay,
                        max_retry_delay=max_retry_delay,
                    )
                    logger.warning(
                        f"[{idx + 1}/{max_retries}] captured error, {retry_delay=}s, {exc=}"
                    )
                    sleep(retry_delay)
            raise prev_exc

        return wrapper

    return decorator


def set_root_cause(exc: Exception, cause: Exception) -> Exception | None:
    while True:
        if exc.__cause__ is None:
            exc.__cause__ = cause
            return exc
        exc = exc.__cause__


def calculate_retry_delay(
    *,
    exc: Exception,
    idx: int,
    initial_retry_delay: float,
    max_retry_delay: float,
) -> float:
    """
    Stolen from https://github.com/openai/openai-python/blob/90aa5eb3ed6b92d9a1de89c0ee063f4768f92256/src/openai/_base_client.py#L586
    """
    
    import openai

    try:
        # About the Retry-After header: https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Retry-After
        #
        # <http-date>". See https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Retry-After#syntax for
        # details.
        if isinstance(exc, openai.APIStatusError) and exc.response.headers:
            retry_header = exc.response.headers.get("retry-after")
            try:
                retry_after = int(retry_header)
            except Exception:
                retry_date_tuple = email.utils.parsedate_tz(retry_header)
                if retry_date_tuple is None:
                    retry_after = -1
                else:
                    retry_date = email.utils.mktime_tz(retry_date_tuple)
                    retry_after = int(retry_date - time.time())
        else:
            retry_after = -1

    except Exception:
        retry_after = -1

    # If the API asks us to wait a certain amount of time (and it's a reasonable amount), just do what it says.
    if 0 < retry_after <= 60:
        return retry_after

    # Apply exponential backoff, but not more than the max.
    sleep_seconds = min(initial_retry_delay * pow(2.0, idx), max_retry_delay)

    # Apply some jitter, plus-or-minus half a second.
    jitter = 1 - 0.25 * random()
    timeout = sleep_seconds * jitter
    return timeout if timeout >= 0 else 0
