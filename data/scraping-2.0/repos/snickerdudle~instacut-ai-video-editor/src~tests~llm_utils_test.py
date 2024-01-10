import unittest
from unittest.mock import patch

from src.utils.llm_utils import OpenAIChat, OpenAIContinuousChat
from src.utils.prompts.prompts import Prompt


class TestOpenAIChat(unittest.TestCase):
    def test_processMessage(self):
        prompt = Prompt(
            prompt_template="You will be given an integer number, like 123, "
            "and your task is to return the number that follows this number ("
            "in this case you would return 124). Return this number and nothing "
            "else.\n\nNUMBER: {self.input_text}\n\nRETURN: ",
        )
        chat = OpenAIChat(prompt=prompt)

        message = "45"
        reply = chat.processMessage(message)
        self.assertEqual(reply, "46")

    def test_getMessageHistory(self):
        chat = OpenAIChat(prompt="Test Prompt")

        message_history = chat.getMessageHistory()

        self.assertEqual(len(message_history), 1)
        self.assertEqual(message_history[0]["role"], "system")
        self.assertEqual(
            message_history[0]["content"], "You are an intelligent assistant"
        )

    def test_getMessageHistoryAfterMessage(self):
        chat = OpenAIChat(prompt="Test Prompt: ")
        chat.processMessage("Test Message")

        message_history = chat.getMessageHistory()

        self.assertEqual(len(message_history), 1)
        self.assertEqual(message_history[0]["role"], "system")
        self.assertEqual(
            message_history[0]["content"], "You are an intelligent assistant"
        )


class TestOpenAIContinuousChat(unittest.TestCase):
    def test_getMessageHistory(self):
        messages = [
            {"role": "user", "content": "User Message 1"},
            {"role": "assistant", "content": "Assistant Response 1"},
            {"role": "user", "content": "User Message 2"},
            {"role": "assistant", "content": "Assistant Response 2"},
        ]
        chat = OpenAIContinuousChat(prompt="Test Prompt", messages=messages)

        message_history = chat.getMessageHistory()

        self.assertEqual(len(message_history), 4)
        self.assertEqual(message_history[0]["role"], "user")
        self.assertEqual(message_history[0]["content"], "User Message 1")
        self.assertEqual(message_history[-1]["role"], "assistant")
        self.assertEqual(
            message_history[-1]["content"], "Assistant Response 2"
        )


if __name__ == "__main__":
    unittest.main()
