import unittest
from unittest.mock import Mock, patch

import openai
import pytest

pytest.skip("no api key", allow_module_level=True)

from jobskills.gpt import gpt  # noqa: E402


class TestGPTFunctions(unittest.TestCase):
    @patch("jobskills.gpt.gpt.openai.ChatCompletion.create")
    def test_job_extract_successful_response(self, mock_create):
        # Mock the API response
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message = {"content": "sample response"}
        mock_response.choices = [mock_choice]
        mock_create.return_value = mock_response

        result = gpt.job_extract("sample prompt")
        self.assertEqual(result, "sample response")

    @patch("jobskills.gpt.gpt.openai.ChatCompletion.create")
    def test_job_compare_successful_response(self, mock_create):
        # Mocking API response again
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message = {"content": "sample response"}
        mock_response.choices = [mock_choice]
        mock_create.return_value = mock_response

        result = gpt.job_extract("sample prompt")
        self.assertEqual(result, "sample response")

    @patch("jobskills.gpt.gpt.openai.ChatCompletion.create")
    def test_job_extract_api_failure_with_retries(self, mock_create):
        # Mock the API to raise an exception every time it's called
        mock_create.side_effect = openai.error.OpenAIError("API Error")

        # script should exit after multiple attempts
        with self.assertRaises(SystemExit):
            gpt.job_extract("sample prompt")

        # api should be called 3 times max
        self.assertEqual(mock_create.call_count, 3)

    @patch("jobskills.gpt.gpt.openai.ChatCompletion.create")
    def test_job_compare_api_failure_with_retries(self, mock_create):
        # Mock the API to raise an exception every time it's called
        mock_create.side_effect = openai.error.OpenAIError("API Error")

        # script should exit after multiple attempts
        with self.assertRaises(SystemExit):
            gpt.job_compare("sample prompt", "fake job list")

        # api should be called 3 times max
        self.assertEqual(mock_create.call_count, 3)
