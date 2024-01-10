# retry_decorator.py

import time
import openai
from functools import wraps

def retry_on_service_unavailable(max_retries=5, backoff_factor=0.5):
    """A decorator for retrying a function call with exponential backoff.

    Args:
        max_retries (int): Maximum number of retries before giving up. Default is 5.
        backoff_factor (float): Multiplier for the delay between retries. Default is 0.5.

    Returns:
        Callable: Decorated function that will be retried on `openai.error.ServiceUnavailableError`.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except openai.error.ServiceUnavailableError:
                    sleep_time = backoff_factor * (2 ** retries)
                    time.sleep(sleep_time)
                    retries += 1
            return func(*args, **kwargs)  # Final attempt, let exception propagate if this fails
        return wrapper
    return decorator
