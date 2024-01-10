import datetime
from unittest.mock import Mock, patch

import pytest
from openai.error import RateLimitError

from superai.data_program.protocol.rate_limit import compute_api_wait_time
from superai.llm.foundation_models.openai import ChatGPT


class OpenAIMockResponse:
    def __init__(self, data):
        self.data = data

    def to_dict_recursive(self):
        return self.data

    def __contains__(self, key):
        return key in self.data

    def __getitem__(self, key):
        if key in self.data:
            item = self.data[key]
            # If the item itself is a dictionary, return a new MockedResponse wrapping that dictionary
            if isinstance(item, dict):
                return OpenAIMockResponse(item)
            else:
                return item
        else:
            raise KeyError(key)


@pytest.fixture()
def chat_gpt_model():
    return ChatGPT()


def test_rate_exceeding_handling(chat_gpt_model):
    chat_gpt_model._wait_for_rate_limits = lambda x, y: True
    with patch("openai.ChatCompletion.create") as chat_mock:
        rate_limit_exception = RateLimitError(
            "Rate limit reached for default-gpt-4 in organization org-XmP3w1BcjkaTrZ6pl7SVUydH on requests per min. Limit: 200 / min. Please try again in 300ms. Contact us through our help center at help.openai.com if you continue to have issues."
        )
        rate_limit_exception.headers = {
            "Date": "Tue, 20 Jun 2023 08:45:54 GMT",
            "Content-Type": "application/json; charset=utf-8",
            "Content-Length": "353",
            "Connection": "keep-alive",
            "vary": "Origin",
            "x-ratelimit-limit-requests": "200",
            "x-ratelimit-remaining-requests": "0",
            "x-ratelimit-reset-requests": "0.65s",
            "x-request-id": "ad79a5edd513dc752cdf540c1f672939",
            "strict-transport-security": "max-age=15724800; includeSubDomains",
            "CF-Cache-Status": "DYNAMIC",
            "Server": "cloudflare",
            "CF-RAY": "7da2bd012f62baee-MXP",
            "alt-svc": 'h3=":443"; ma=86400',
        }
        chat_mock.side_effect = [
            rate_limit_exception,
            OpenAIMockResponse({"choices": [{"message": {"content": "The capital of Jordan is Amman"}}]}),
        ]
        result = chat_gpt_model.predict("what's the capital of Jordan?")
        assert result


def test_wait_for_rate_limits(monkeypatch, chat_gpt_model):
    # RPM call, retrial, TPM call, RPM retrial, TPM retrial
    return_data = [0.5, 0, 0.1, 0, 0]
    test_mock = Mock(side_effect=return_data)
    monkeypatch.setattr(
        "superai.llm.foundation_models.openai.compute_api_wait_time",
        test_mock,
    )

    chat_gpt_model._wait_for_rate_limits("gpt-3.5-turbo", 50)
    assert test_mock.call_count == len(return_data)


@patch("superai.data_program.protocol.rate_limit.datetime")
def test_compute_api_wait_time(datetime_mock):
    model_name = "fancy_model"
    passed_secs = 3

    # Fixes time, to make tests consistent
    datetime_mock.datetime.now = Mock(return_value=datetime.datetime(2023, 2, 26, 0, 10, passed_secs, 0))

    # First call, tests fresh key
    assert compute_api_wait_time(model_name, 30, 25) == 0
    # Seconds call should't exceed threshold
    assert compute_api_wait_time(model_name, 30, 1) == 0
    # Third call should exceed.
    assert compute_api_wait_time(model_name, 30, 10) == 60 - passed_secs
    # New model call shouldn't exceed
    assert compute_api_wait_time(model_name + "_NEW", 30, 10) == 0
    # New minute should reset rates
    datetime_mock.datetime.now = Mock(return_value=datetime.datetime(2023, 2, 26, 0, 11, passed_secs, 0))
    assert compute_api_wait_time(model_name, 30, 1) == 0
