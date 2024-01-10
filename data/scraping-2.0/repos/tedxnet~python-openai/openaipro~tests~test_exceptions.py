import pickle

import pytest

import openai

EXCEPTION_TEST_CASES = [
    openaipro.InvalidRequestError(
        "message",
        "param",
        code=400,
        http_body={"test": "test1"},
        http_status="fail",
        json_body={"text": "iono some text"},
        headers={"request-id": "asasd"},
    ),
    openaipro.error.AuthenticationError(),
    openaipro.error.PermissionError(),
    openaipro.error.RateLimitError(),
    openaipro.error.ServiceUnavailableError(),
    openaipro.error.SignatureVerificationError("message", "sig_header?"),
    openaipro.error.APIConnectionError("message!", should_retry=True),
    openaipro.error.TryAgain(),
    openaipro.error.Timeout(),
    openaipro.error.APIError(
        message="message",
        code=400,
        http_body={"test": "test1"},
        http_status="fail",
        json_body={"text": "iono some text"},
        headers={"request-id": "asasd"},
    ),
    openaipro.error.OpenAIError(),
]


class TestExceptions:
    @pytest.mark.parametrize("error", EXCEPTION_TEST_CASES)
    def test_exceptions_are_pickleable(self, error) -> None:
        assert error.__repr__() == pickle.loads(pickle.dumps(error)).__repr__()
