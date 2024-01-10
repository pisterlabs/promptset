import logging
from functools import wraps

from aidial_sdk import HTTPException
from openai import OpenAIError

logger = logging.getLogger(__name__)


class RequestParameterValidationError(Exception):
    def __init__(self, message: str, param: str, *args: object) -> None:
        super().__init__(message, *args)
        self._param = param

    @property
    def param(self) -> str:
        return self._param


def _to_http_exception(e: Exception) -> HTTPException:
    if isinstance(e, RequestParameterValidationError):
        return HTTPException(
            message=str(e),
            status_code=422,
            type="invalid_request_error",
            param=e.param,
        )

    if isinstance(e, OpenAIError):
        http_status = e.http_status or 500
        if e.error:
            return HTTPException(
                message=e.error.message,
                status_code=http_status,
                type=e.error.type,
                code=e.error.code,
                param=e.error.param,
            )

        return HTTPException(message=str(e), status_code=http_status)

    return HTTPException(
        message=str(e), status_code=500, type="internal_server_error"
    )


def unhandled_exception_handler(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.exception("Unhandled exception")
            raise _to_http_exception(e)

    return wrapper
