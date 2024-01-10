import pytest

import openai

mock_completition_data = {
    "id": "mockid",
    "object": "chat.completion",
    "created": 1684779054,
    "model": "gpt-5",
    "usage": {"prompt_tokens": 25, "completion_tokens": 1, "total_tokens": 26},
    "choices": [
        {
            "message": {"role": "assistant", "content": "MOCKED!", "name": "n"},
            "finish_reason": "stop",
            "index": 0,
        }
    ],
}


@pytest.fixture()
def mock_openai(monkeypatch, request):
    def mock_create(*args, **kwargs):
        if hasattr(request, "param"):
            return request.param
        else:
            return mock_completition_data

    async def amock_create(*args, **kwargs):
        if hasattr(request, "param"):
            return request.param
        else:
            return mock_completition_data

    monkeypatch.setattr(
        openai.ChatCompletion,
        "create",
        mock_create,
    )

    monkeypatch.setattr(
        openai.ChatCompletion,
        "acreate",
        amock_create,
    )


@pytest.fixture()
def mock_openai_exception_factory(monkeypatch, request):
    """Factory for mocking OpenAI exceptions with different exceptions in tests Use like this:
    `mock_openai_exception_factory(openai.error.Timeout("mocked timeout")])`"""

    def _mock_openai_exception(request):
        def mock_create(*args, **kwargs):
            raise request

        async def amock_create(*args, **kwargs):
            raise request

        monkeypatch.setattr(
            openai.ChatCompletion,
            "create",
            mock_create,
        )

        monkeypatch.setattr(
            openai.ChatCompletion,
            "acreate",
            amock_create,
        )

    return _mock_openai_exception
