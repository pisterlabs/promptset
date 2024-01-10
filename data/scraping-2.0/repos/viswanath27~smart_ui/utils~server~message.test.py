import unittest
from unittest import TestCase
from typing import List

from .message import createMessagesToSend
from .tiktoken import getTiktokenEncoding
from .types import Message
from .types.openai import OpenAIModel


class CreateMessagesToSendTestCase(TestCase):
    def test_create_messages_to_send(self):
        encoding = getTiktokenEncoding('gpt-3.5-turbo')
        systemPrompt = 'Hello'
        model: OpenAIModel = {
            'id': 'gpt-3.5-turbo',
            'name': 'gpt-3.5-turbo',
            'tokenLimit': 1100,
            'maxLength': 4000,
        }
        messages: List[Message] = [
            {'role': 'user', 'content': 'World'},
            {'role': 'assistant', 'content': 'How are you?'},
            {'role': 'user', 'content': 'Fine, thank you.'},
        ]

        result = createMessagesToSend(
            encoding,
            model,
            systemPrompt,
            100,
            messages,
        )

        self.assertEqual(result['messages'][0], {'role': 'user', 'content': 'World'})
        self.assertEqual(result['maxToken'], 1066)


if __name__ == '__main__':
    unittest.main()

