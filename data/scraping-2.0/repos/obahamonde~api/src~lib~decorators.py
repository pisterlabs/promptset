"""
Decorators for the library.
"""
import asyncio
import functools
import logging
from time import perf_counter
from typing import Awaitable, Callable, Generator, Sequence, TypeVar, Union

from fastapi import HTTPException, status
from openai import (
    APIConnectionError,
    APIError,
    APIResponseValidationError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    ConflictError,
    NotFoundError,
    OpenAIError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
)
from rich.console import Console
from rich.logging import RichHandler
from rich.pretty import install
from rich.traceback import install as ins
from tenacity import retry as retry_
from tenacity import retry_if_exception_type, stop_after_attempt, wait_exponential
from typing_extensions import ParamSpec

ERRORS = (
    asyncio.TimeoutError,
    ConnectionError,
    ConnectionRefusedError,
    ConnectionResetError,
    TimeoutError,
    UnicodeDecodeError,
    UnicodeEncodeError,
    UnicodeError,
    TypeError,
    ValueError,
    ZeroDivisionError,
    IndexError,
    AttributeError,
    ImportError,
    ModuleNotFoundError,
    NotImplementedError,
    RecursionError,
    OverflowError,
    KeyError,
    Exception,
)

OPEN_AI_HTTP_EXCEPTIONS = (
    APIResponseValidationError,
    APIStatusError,
    RateLimitError,
    AuthenticationError,
    BadRequestError,
    ConflictError,
    NotFoundError,
    PermissionDeniedError,
    UnprocessableEntityError,
)

OPEN_AI_ERRORS = (APIConnectionError, APIError, APITimeoutError, OpenAIError)

RETRY_EXCEPTIONS = (
    asyncio.TimeoutError,
    ConnectionError,
    ConnectionRefusedError,
    ConnectionResetError,
    TimeoutError,
    APIConnectionError,
    APITimeoutError,
    APIError,
)

T = TypeVar("T")
P = ParamSpec("P")


def setup_logging(name: str = __name__) -> logging.Logger:
    """
    Set's up logging using the Rich library for pretty and informative terminal logs.

    Arguments:
    name -- Name for the logger instance. It's best practice to use the name of the module where logger is defined.
    """
    install()
    ins()
    console = Console(record=True, force_terminal=True)
    console_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=True,
        markup=True,
        rich_tracebacks=True,
        tracebacks_show_locals=True,
        tracebacks_extra_lines=2,
        tracebacks_theme="monokai",
        show_level=False,
    )
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    console_handler.setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO, handlers=[console_handler])
    logger_ = logging.getLogger(name)
    logger_.setLevel(logging.INFO)
    return logger_


logger = setup_logging()


def process_time(
    func: Callable[P, Union[Awaitable[T], T]]
) -> Callable[P, Awaitable[T]]:
    """
    A decorator to measure the execution time of a coroutine.

    Arguments:
    func -- The coroutine whose execution time is to be measured.
    """

    @functools.wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        """
        Wrapper function to time the function call.
        """
        start = perf_counter()
        if asyncio.iscoroutinefunction(func):
            result = await func(*args, **kwargs)
        else:
            result = func(*args, **kwargs)
        end = perf_counter()
        logger.info(
            "Time taken to execute %s: %s seconds", wrapper.__name__, end - start
        )
        return result  # type: ignore

    return wrapper


def handle_errors(
    func: Callable[P, Union[Awaitable[T], T]]
) -> Callable[P, Awaitable[T]]:
    """
    A decorator to handle errors in a coroutine.

    Arguments:
    func -- The coroutine whose errors are to be handled.
    """

    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        try:
            logger.info("Calling %s", func.__name__)
            logger.info("Args: %s", args)
            logger.info("Kwargs: %s", kwargs)
            if asyncio.iscoroutinefunction(func):
                response = await func(*args, **kwargs)
                logger.info(response)
                return response  # type: ignore
            response = func(*args, **kwargs)
            logger.info(response)
            return response  # type: ignore
        except HTTPException as exc:
            logger.error("HTTPException: %s %s", exc.status_code, exc.detail)
            raise HTTPException(detail=exc.detail, status_code=exc.status_code) from exc
        except OPEN_AI_HTTP_EXCEPTIONS as exc:
            logger.error("OpenAI HTTPException: %s %s", exc.message, exc.status_code)
            raise HTTPException(
                detail=exc.message, status_code=exc.status_code
            ) from exc
        except OPEN_AI_ERRORS as exc:
            logger.error("OpenAI Error: %s %s", exc.__class__.__name__, exc)
            raise HTTPException(
                detail=str(exc), status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
            ) from exc
        except ERRORS as exc:
            logger.error("Error: %s %s", exc.__class__.__name__, exc)
            raise HTTPException(
                detail=str(exc), status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
            ) from exc

    return wrapper


def chunker(seq: Sequence[T], size: int) -> Generator[Sequence[T], None, None]:
    """
    A generator function that chunks a sequence into smaller sequences of the given size.

    Arguments:
    seq -- The sequence to be chunked.
    size -- The size of the chunks.
    """
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))


def gen_emptystr() -> str:
    """
    A generator function that returns an empty string.
    """
    return ""


def handle(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
    """
    A decorator to apply all decorators to a coroutine.

    Arguments:

    func -- The coroutine to decorate.
    """
    return functools.reduce(
        lambda f, g: g(f),  # type: ignore
        [retry(), handle_errors, process_time],
        func,
    )


def async_io(func: Callable[P, T]) -> Callable[P, Awaitable[T]]:
    """
    Decorator to convert an IO bound function to a coroutine by running it in a thread pool.
    """

    @handle
    @functools.wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        return await asyncio.to_thread(func, *args, **kwargs)

    return wrapper


def async_cpu(func: Callable[P, T]) -> Callable[P, Awaitable[T]]:
    """
    Decorator to convert a CPU bound function to a coroutine by running it in a process pool.
    """

    @handle
    @functools.wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        return await asyncio.get_running_loop().run_in_executor(
            None, func, *args, **kwargs
        )

    return wrapper


def retry(
    retries: int = 10, wait: int = 1, max_wait: int = 10
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """Wrap an async function with exponential backoff."""

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @functools.wraps(func)
        @retry_(
            stop=stop_after_attempt(retries),
            wait=wait_exponential(multiplier=wait, max=max_wait),
            retry=retry_if_exception_type(RETRY_EXCEPTIONS),
            reraise=True,
        )
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            return await func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator
