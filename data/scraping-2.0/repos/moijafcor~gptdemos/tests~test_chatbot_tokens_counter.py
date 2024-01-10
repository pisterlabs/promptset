import unittest
from unittest.mock import patch
from openai_chatbot_tokens_counter import main


class TestChatbotTokensCounter(unittest.TestCase):
    @patch(
        "builtins.input", side_effect=["Hello", "How are you?", "I'm fine, thank you!"]
    )
    def test_chatbot_tokens_counter(self, mock_input):
        main()
        # Add your assertions here


if __name__ == "__main__":
    unittest.main()
