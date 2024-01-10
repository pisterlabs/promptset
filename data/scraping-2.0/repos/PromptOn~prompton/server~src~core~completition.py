import asyncio
import time
import openai

from src.schemas.openAI import (
    ChatGPTChatCompletitionRequest,
    ChatGPTChatCompletitionResponse,
)
from src.schemas.inference import (
    InferenceError,
    InferenceResponseData,
    InferenceResponseError,
)
from src.core.settings import settings


MOCK_COMPLETITION = {
    "id": "mockid",
    "object": "chat.completion",
    "created": 1684779054,
    "model": "gpt-5",
    "usage": {"prompt_tokens": 25, "completion_tokens": 1, "total_tokens": 26},
    "choices": [
        {
            "message": {
                "role": "assistant",
                "content": "This is a mocked response - called with 'mock_me_softly'",
            },
            "finish_reason": "stop",
            "index": 0,
        }
    ],
}


async def _completition_task(
    raw_request: ChatGPTChatCompletitionRequest, request_timeout: float
):
    """It needs to be a task so we can capture the timeout (asyncio.CancelledError) exception from acreate"""
    completition = None

    # NB: these are for manual testing, unit tests are using mocks so this can be safely removed
    if raw_request.user == "fail_me_softly":
        await asyncio.sleep(10)
        raise openai.APIError("mocking error")
    elif raw_request.user == "timeout_me_softly":
        await asyncio.sleep(10)
        raise openai.error.Timeout(  # pyright: ignore[reportGeneralTypeIssues]
            "timeout error"
        )
    elif raw_request.user == "mock_me_softly":
        await asyncio.sleep(10)
        completition = MOCK_COMPLETITION
    else:
        completition = await openai.ChatCompletion.acreate(
            **raw_request.dict(exclude_none=True), request_timeout=request_timeout
        )

    raw_response = ChatGPTChatCompletitionResponse.parse_obj(completition)

    return raw_response


async def get_openai_chat_completition(
    raw_request: ChatGPTChatCompletitionRequest,
    openai_api_key: str,
    *,
    request_timeout: float | None,
):
    if request_timeout is None:
        request_timeout = settings.DEFAULT_OPENAI_REQUEST_TIMEOUT_SECONDS

    openai.api_key = openai_api_key

    start_time = time.perf_counter()

    response: InferenceResponseData | InferenceResponseError

    try:
        task = asyncio.create_task(_completition_task(raw_request, request_timeout))

        raw_response = await task

        response = InferenceResponseData(
            token_usage=raw_response.usage,
            raw_response=raw_response,
        )

    except Exception as e:
        full_classname = f"{e.__class__.__module__}.{e.__class__.__name__}"
        response = InferenceResponseError(
            error=InferenceError(
                error_class=full_classname, message=str(e), details={**vars(e)}
            )
        )
    finally:
        end_time = time.perf_counter()

    response.completition_duration_seconds = round(end_time - start_time, 4)

    return response
