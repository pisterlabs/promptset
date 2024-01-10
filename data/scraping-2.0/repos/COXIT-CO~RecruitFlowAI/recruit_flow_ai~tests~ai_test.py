"""
This module contains unit tests for the RecruitFlowAI class. 
"""
import unittest
from unittest.mock import create_autospec, patch

import httpx
from recruit_flow_ai import RecruitFlowAI
import openai
from openai.types.chat.chat_completion import ChatCompletion


class TestRecruitFlowAI(unittest.TestCase):
    """
    Unit test class for testing the RecruitFlowAI class.
    """

    def setUp(self):
        self.ai = RecruitFlowAI(api_key="sk-test12345678901234567890123456789012")

    def test_is_valid_api_key_format(self):
        self.assertTrue(
            self.ai.is_valid_api_key_format("sk-test12345678901234567890123456789012")
        )
        self.assertFalse(self.ai.is_valid_api_key_format("invalid_key"))

    def test_is_valid_temperature(self):
        self.assertTrue(self.ai.is_valid_temperature(0.5))
        self.assertFalse(self.ai.is_valid_temperature(1.5))

    @patch("openai.chat.completions.create")
    def test_generate_response(self, mock_create):
        mock_create.return_value = ChatCompletion(
            id="chatcmpl-7QyqpwdfhqwajicIEznoc6Q47XAyW",
            choices=[
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {
                        "content": "Test response",
                        "role": "assistant",
                    },
                }
            ],
            created=1677664795,
            model="gpt-3.5-turbo-0613",
            object="chat.completion",
            usage={
                "completion_tokens": 17,
                "prompt_tokens": 57,
                "total_tokens": 74,
            },
        )
        response = self.ai.generate_response(
            openai_msgs=[{"role": "user", "content": "Test message"}]
        )
        self.assertEqual(response, "Test response")

    @patch("openai.chat.completions.create")
    def test_generate_response_error(self, mock_create):
        mock_request = create_autospec(httpx.Request, instance=True)
        mock_create.side_effect = openai.APIError(
            message="Error message", request=mock_request, body=None
        )
        response = self.ai.generate_response(
            openai_msgs=[{"role": "user", "content": "Test message"}]
        )
        self.assertIn("Report this to #recruitflowai_issues channel.", response)


if __name__ == "__main__":
    unittest.main()
