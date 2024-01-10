import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import unittest
from unittest.mock import MagicMock
from openai_client import OpenAIClient
from leetcode_processor import LeetCodeProcessor


class TestLeetCodeProcessor(unittest.TestCase):
    def setUp(self):
        self.mock_client = MagicMock(spec=OpenAIClient)
        self.processor = LeetCodeProcessor(self.mock_client)

    def test_init(self):
        self.assertIsInstance(self.processor.client, OpenAIClient)

    def test_pretty_print_with_valid_messages(self):
        messages = [
            {"role": "assistant", "content": [{"text": {"value": "test message"}}]}
        ]
        result = self.processor.pretty_print(messages)
        self.assertEqual(result, "test message")

    def test_pretty_print_with_empty_messages(self):
        result = self.processor.pretty_print([])
        self.assertIsNone(result)

    def test_pretty_print_with_no_assistant_role(self):
        messages = [{"role": "user", "content": [{"text": {"value": "test message"}}]}]
        result = self.processor.pretty_print(messages)
        self.assertIsNone(result)

    def test_pretty_print_with_invalid_messages(self):
        messages = [{"role": "assistant"}]  # No 'content' field
        with self.assertRaises(Exception):
            self.processor.pretty_print(messages)

    def test_process_problem_with_valid_problem_number(self):
        self.mock_client.create_thread_and_run.return_value = ("thread_id", "run_id")
        self.mock_client.wait_on_run.return_value = "run_id"
        self.mock_client.get_response.return_value = [
            {"role": "assistant", "content": [{"text": {"value": "Problem solved"}}]}
        ]

        problem_number, output = self.processor.process_problem(1)
        self.assertEqual(problem_number, 1)
        self.assertEqual(output, "Problem solved")

    def test_process_problem_with_invalid_problem_number(self):
        problem_number, output = self.processor.process_problem(-1)
        self.assertIsNone(output)

    def test_process_problem_with_exception(self):
        self.mock_client.create_thread_and_run.side_effect = Exception("Error")

        problem_number, output = self.processor.process_problem(1)
        self.assertIsNone(output)


if __name__ == "__main__":
    unittest.main()
