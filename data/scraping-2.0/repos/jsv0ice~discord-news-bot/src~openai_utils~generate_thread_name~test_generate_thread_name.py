import unittest
import openai
from unittest.mock import patch
from .generate_thread_name import generate_thread_name

# An example class to mimic the behavior of pycord's Embed
class Embed:
    def __init__(self, description):
        self.description = description

# An example class to mimic the behavior of pycord's File
class File:
    def __init__(self, filename):
        self.filename = filename

class TestGenerateThreadName(unittest.IsolatedAsyncioTestCase):
    @patch.object(openai.ChatCompletion, 'create')
    async def test_generate_thread_name(self, mock_chat_completion):
        content = "Test content"
        embeds = [Embed(description="Test description")]

        # Create File objects with filename attribute
        files = [File(filename="file1.txt"), File(filename="file2.txt")]
        
        expected_response = "Generated Thread Name"

        mock_chat_completion.return_value = {
            "choices": [
                {
                    "message": {
                        "content": expected_response
                    }
                }
            ]
        }

        result = await generate_thread_name(content, embeds, files)

        self.assertEqual(result, expected_response)

if __name__ == '__main__':
    unittest.main()