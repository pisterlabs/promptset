import functools
import time
import asyncio

import openai

from moview.config.loggers.mongo_logger import error_logger
from moview.exception.retry_execution_error import RetryExecutionError


def retry(max_retries=3, retry_delay=2):
    """
    재시도 횟수를 초과할 때까지 함수를 재시도하는 데코레이터입니다.
    Args:
        max_retries: 최대 반복 횟수
        retry_delay: 실패했을 때 대기 시간

    Returns:
        재시도 횟수를 초과할 때까지 함수를 재시도하는 데코레이터
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0

            while retries < max_retries:
                try:
                    result = func(*args, **kwargs)
                    return result  # 성공한 경우 결과를 반환합니다.

                except openai.error.RateLimitError as e:  # 토큰 사용량이 초과되면, 예외 떠넘김
                    raise e

                except Exception as e:
                    retries += 1
                    if retries < max_retries:
                        error_logger(msg=f"{func.__name__} Execution Error: Retry ({retries}/{max_retries})...",
                                     error=str(e))
                        time.sleep(retry_delay)
                    else:
                        error_logger(msg=f"{func.__name__} Execution Error: Failed due to excessive number of retries.",
                                     error=str(e))
                        raise RetryExecutionError()  # 재시도 횟수를 초과하면 예외를 다시 발생시킵니다.

        return wrapper

    return decorator


def async_retry(max_retries=3, retry_delay=2):
    """
    재시도 횟수를 초과할 때까지 함수를 재시도하는 데코레이터입니다.
    Args:
        max_retries: 최대 반복 횟수
        retry_delay: 실패했을 때 대기 시간

    Returns:
        재시도 횟수를 초과할 때까지 함수를 재시도하는 데코레이터
    """

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            retries = 0

            while retries < max_retries:
                try:
                    result = await func(*args, **kwargs)
                    return result  # 성공한 경우 결과를 반환합니다.

                except openai.error.RateLimitError as e:  # 토큰 사용량이 초과되면, 예외 떠넘김
                    raise e

                except Exception as e:
                    retries += 1
                    if retries < max_retries:
                        error_logger(msg=f"{func.__name__} Execution Error: Retry ({retries}/{max_retries})...",
                                     error=str(e))
                        await asyncio.sleep(retry_delay)  # 비동기 대기
                    else:
                        error_logger(msg=f"{func.__name__} Execution Error: Failed due to excessive number of retries.",
                                     error=str(e))
                        raise RetryExecutionError()  # 재시도 횟수를 초과하면 예외를 다시 발생시킵니다.

        return wrapper

    return decorator
