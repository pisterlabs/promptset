import os, sys, unittest, openai

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from src.utils import openai_completion
from unittest.mock import patch, MagicMock

response = {
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "message": {
        "content": "The 2020 World Series was played in Texas at Globe Life Field in Arlington.",
        "role": "assistant"
      }
    }
  ]
}

class TestUtils(unittest.TestCase):
    @patch("openai.ChatCompletion.create")
    def test_openai_completion(self, mock_openai_chatcompletion):
        mock_openai_chatcompletion.return_value = response
        rtn_item = openai_completion(input, 1)
        self.assertEqual(rtn_item, ["The 2020 World Series was played in Texas at Globe Life Field in Arlington."])
        

