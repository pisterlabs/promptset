import openai
import unittest
from unittest.mock import patch

import scriptgpt

# Initialize the OpenAI client
client = openai.OpenAI(api_key="dummy key")

class Delta:
    def __init__(self, content):
        self.content = content

class Choice:
    def __init__(self, content):
        self.delta = Delta(content)

class Chunk:
    def __init__(self, choice):
        self.choices = [choice]

class TestScriptGPT(unittest.TestCase):
    @patch.object(client.chat.completions, 'create')
    def test_scriptgpt_response(self, mock_create):
        # Arrange
        user_input = 'Test input'
        mock_create.return_value = [Chunk(Choice('Test response'))]

        # Call the function to test
        stream = scriptgpt.scriptgpt_response(user_input, client)

        # Check that the response from the function matches the mocked response
        for chunk in stream: 
            chunkContent = chunk.choices[0].delta.content 
        self.assertEqual(chunkContent, 'Test response')


    @patch('builtins.input', return_value='quit')
    @patch('scriptgpt.scriptgpt_response', return_value='Goodbye! Have a nice day!')
    def test_chat(self, input_mock, response_mock):
        with patch('builtins.print') as print_mock:
            scriptgpt.chat()
            print_mock.assert_called_with('\033[94mScriptGPT: \033[0mGoodbye! Have a nice day!')

if __name__ == '__main__':
    unittest.main()