"""
Tests for the LLM Invokers Module
---------------------------------

This module contains tests for the LLM Invokers, specifically focusing on the OpenAIInvoker.

Classes:
    - TestOpenAIInvoker: Tests for the OpenAIInvoker class.
"""

import unittest
from unittest.mock import patch

# pylint: disable=import-error
from langchain.schema import AIMessage

from src.llm.openai_invoker import OpenAIInvoker


class TestOpenAIInvoker(unittest.TestCase):
    """
    Test cases for the OpenAIInvoker class.
    """

    def setUp(self):
        self.invoker = OpenAIInvoker()

    @patch("src.llm.openai_invoker.ChatOpenAI.predict_messages")
    def test_invoke(self, mock_predict_messages):
        """
        Test the invoke method of the OpenAIInvoker.
        """
        # Mock the response from the OpenAIChat's predict_messages method
        mock_response = AIMessage(content="Mocked response for: Hello, how are you?")
        mock_predict_messages.return_value = mock_response

        # Test the invoke method
        prompt = "Hello, how are you?"
        expected_response = "Mocked response for: Hello, how are you?"
        actual_response = self.invoker.invoke(prompt)

        self.assertEqual(actual_response, expected_response)


class TestOpenAIInvokerIntegration(unittest.TestCase):
    """
    Integration tests for the OpenAIInvoker class.
    """

    def setUp(self):
        self.invoker = OpenAIInvoker()

    def test_invoke_integration(self):
        """
        Integration test for the invoke method of the OpenAIInvoker.
        This test actually calls the OpenAI API and checks the response.
        """
        # Define a sample prompt
        prompt = "Translate the following English text to French: 'Hello, how are you?'"

        # Invoke the OpenAI model with the given prompt
        response = self.invoker.invoke(prompt)

        # Check if the response is not empty (since the actual translation might vary)
        self.assertIsNotNone(response)
        self.assertNotEqual(response, "")


if __name__ == "__main__":
    unittest.main()
