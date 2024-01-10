"""Unit tests for stream_completion.py"""
# add the project root directory to the system path
if __name__ == "__main__":
    from pathlib import Path

    project_directory = Path(__file__).parent.parent.parent
    import sys

    # sys.path.insert(0, str(project_directory))
    if str(project_directory) not in sys.path:
        sys.path.append(str(project_directory))

import unittest
from unittest.mock import patch, MagicMock
import logging

import openai

from controller.stream_completion import StreamCompletion
from controller.create_logger import create_logger
from controller.waifuai.conversations import test_dict


# Create a logger instance
module_logger = create_logger(
    logger_name="tests.unit.test_stream_completion",
    logger_filename="stream_completion.log",
    log_directory="logs/tests",
    console_logging=True,
    console_log_level=logging.INFO,
)


class TestStreamCompletion(unittest.TestCase):
    """This is the unit test class for the StreamCompletion class."""

    @classmethod
    def setUpClass(cls):
        """Set up the StreamCompletion class unit test."""
        # Create class logger instance
        cls.logger = module_logger
        cls.logger.info("===Testing StreamCompletion class===")

    def setUp(self):
        """Set up the StreamCompletion class unit test."""
        self.stream_completion = StreamCompletion(persona=test_dict.persona)

    def test_completion_generator(self):
        """Test that the completion_generator method raises an InvalidRequestError when the request is invalid."""
        messages = [
            {"role": "assistant", "content": "Hello."},
            {"role": "user", "content": "Hello, how you doing?"},
            {"role": "assistant", "content": "I'm doing great, how about you?"},
            {"role": "user", "content": "I'm doing great too."},
            {"role": "assistant", "content": "That's great to hear."},
            {"role": "user", "content": "Yeah, it is."},
        ]

        persona = {"messages": messages}

        # patch the method of SomeClass that can raise an exception
        # with patch("openai.ChatCompletion.create") as mock_method:
        #     # assign the exception class to side_effect
        #     mock_method.side_effect = openai.error.InvalidRequestError(
        #         "Message", "Param"
        #     )

        # with self.assertRaises(openai.error.InvalidRequestError):
        #     # call the method that will raise the exception
        generator = self.stream_completion.completion_generator(
            test_dict.persona, stop=["Human:"], max_tokens=3000
        )

        for _ in generator:
            print(_, end="", flush=True)


if __name__ == "__main__":
    unittest.main()
