import unittest
import base64
import json
from typing import List
from src.data_handler import DataHandler
from openai.openai_response import OpenAIResponse
from src.generation.openai_text import OpenAITextGeneration
from src.messages.context import ContextWindow


class TestCreateMessage(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.openai_text = OpenAITextGeneration()
        self.handler = DataHandler(persona_image="src/static/images/Eris0001.png")
        self.context = self.handler.context

    def test_handle_message(self):
        message = self.handler.create_messages.create_message("user", "Write me a haiku about a duck wearing a fedora")
        responses = list(self.openai_text.send_chat_complete(messages=[json.loads(message)]))
        full_response = ""
        for response in responses:
            full_response += response.choices[0].delta.get("content", "")
        self.assertTrue(isinstance(full_response, str))


    def test_handle_image(self):
        result = self.handler.handle_image()
        with open(self.handler.image_path, "rb") as f:
            expected_result = base64.b64encode(f.read()).decode("utf-8")
        self.assertEqual(result, expected_result)