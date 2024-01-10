"""Test the utils.
Run test: poetry run pytest wenling/tests/common/test_model_utils.py
"""
import os
from unittest.mock import Mock, patch

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from wenling.common.model_utils import OpenAIChatModel


class TestOpenAIChatModel:
    @pytest.fixture
    def model(self):
        # Setup for the test, instantiate your model
        return OpenAIChatModel()

    @pytest.mark.parametrize(
        "model_type, sys_prompt, user_prompt, max_tokens, temperature, expected_result",
        [
            (
                "gpt-3.5-turbo-1106",
                "Please summarize the following:",
                "Hello, my name is John and I am 30 years old.",
                50,
                0.0,
                "John is a 30 year old.",
            ),
            (
                "gpt-3.5-turbo-1106",
                "Please generate a response:",
                "How are you?",
                20,
                0.5,
                "I'm doing well, thanks for asking!",
            ),
        ],
    )
    def test_inference_success(self, model_type, sys_prompt, user_prompt, max_tokens, temperature, expected_result):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "mocked_api_key"}):
            # Mock the OpenAI client
            with patch("openai.OpenAI") as MockOpenAI:
                mock_openai_instance = MockOpenAI.return_value

                # Set up the mock for chat.completions.create method
                mock_openai_instance.chat.completions.create.return_value = ChatCompletion(
                    id="chatcmpl-123",
                    created=1677652288,
                    model="gpt-3.5-turbo-1106",
                    object="chat.completion",
                    choices=[
                        Choice(
                            message=ChatCompletionMessage(role="assistant", content=expected_result),
                            finish_reason="stop",
                            index=0,
                            text="aaa",
                            logprobs={"content": []},
                        )
                    ],
                )

                # Create an instance of OpenAIChatModel and call the inference method
                model = OpenAIChatModel()
                result = model.inference(
                    user_prompt=user_prompt,
                    sys_prompt=sys_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    model_type=model_type,
                )

                assert result == expected_result

    def test_inference_no_choices(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "mocked_api_key"}):
            # Mock the OpenAI client
            with patch("openai.OpenAI") as MockOpenAI:
                model = OpenAIChatModel()
                # Mocking the OpenAI client's behavior for a response with no choices
                mock_openai_instance = MockOpenAI.return_value
                mock_openai_instance.chat.completions.create.return_value = ChatCompletion(
                    id="chatcmpl-123",
                    created=1677652288,
                    model="gpt-3.5-turbo-1106",
                    object="chat.completion",
                    choices=[],
                )

                # Expect an exception when choices are empty
                with pytest.raises(Exception) as excinfo:
                    model.inference(user_prompt="Hello")
                assert "Failed to parse choices" in str(excinfo.value)
