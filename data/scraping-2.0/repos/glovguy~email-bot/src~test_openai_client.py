import unittest
from unittest.mock import patch
from openai_client import OpenAIClient

class OpenAIClientTest(unittest.TestCase):

    @patch('openai_client.openai')
    def test_send_message_to_openai(self, mock_openai):
        mock_openai.ChatCompletion.create.return_value = {'choices': [{'message': {'content': 'Test Response'}}]}

        client = OpenAIClient()
        response = client.send_message([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, OpenAI!"}
        ])

        self.assertEqual(response, { "content": "Test Response" })
