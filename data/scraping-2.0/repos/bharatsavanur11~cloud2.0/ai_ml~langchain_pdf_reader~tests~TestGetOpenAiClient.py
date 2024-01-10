import unittest


import unittest
from unittest.mock import patch, mock_open
import langchain_pdf_reader as reader  # replace 'your_module' with the name of the module containing getOpenAiClient function

class TestGetOpenAiClient(unittest.TestCase):

    @patch('builtins.open', new_callable=mock_open, read_data='fake_api_key\n')
    def test_getOpenAiClient(self, mock_file):
        client = reader.getOpenAiClient()  # replace 'your_module' with the name of the module containing getOpenAiClient function
        self.assertIsNotNone(client)
        mock_file.assert_called_with('api_key')

if __name__ == '__main__':
    unittest.main()