"""test_finetune.py Unit testing of finetune.py"""

from unittest.mock import patch

import openai
from openai.openai_object import OpenAIObject

from clozify_llm.constants import STARTING_MESSAGE
from clozify_llm.predict import ChatCompleter, Completer, GenericCompleter


class TestCompleter:
    @patch("clozify_llm.predict.openai.Completion")
    def test_completer_get_completion_response(self, mock_completion, completion_response):
        """Test Completer.get_completion_response() call with mocked OpenAI Completion response"""
        mock_completion.create.return_value = completion_response
        completer = Completer("my_model_id")
        result = completer.get_completion_response("dummy_word", "dummy_defn")

        expected = completion_response

        assert result == expected

    def test_extract_text_from_response(self, completion_response):
        """ "Test Completer.extract_text_from_response() on simple example"""
        completer = Completer("my_model_id")
        result = completer.extract_text_from_response(completion_response)
        expected = "This is indeed a test"
        assert result == expected


class TestChatCompleter:
    @patch("clozify_llm.predict.openai.ChatCompletion")
    def test_completer_get_completion_response(self, mock_chat_completion, chat_completion_response):
        """Test ChatCompleter.get_completion_response() call with mocked OpenAI ChatCompletion response"""
        mock_chat_completion.create.return_value = chat_completion_response
        completer = ChatCompleter("chat_model_id")
        result = completer.get_completion_response("dummy_word", "dummy_defn")

        expected = chat_completion_response

        assert result == expected

    def test_extract_text_from_response(self, chat_completion_response):
        """ "Test ChatCompleter.extract_text_from_response() on simple example"""
        completer = ChatCompleter()
        result = completer.extract_text_from_response(chat_completion_response)
        expected = "Hello there, how may I assist you today?"
        assert result == expected

    def test_make_chat_params(self):
        """Test ChatCompleter._make_chat_params() on simple example"""
        completer = ChatCompleter(model_id="my_chat_model")
        result = completer._make_chat_params(input_word="word", temperature=0.9, max_tokens=100)
        expected_messages = STARTING_MESSAGE + [{"role": "user", "content": "Input: word"}]
        expected = {
            "model": "my_chat_model",
            "temperature": 0.9,
            "max_tokens": 100,
            "messages": expected_messages,
        }
        assert result == expected


class DummyCompleter(GenericCompleter):
    """Dummy subclass of GenericCompleter for testing"""

    def __init__(self, model_id: str):
        super().__init__(openai_resource=openai.Completion, model_id=model_id)

    def get_completion_response(self, word, defn):
        return OpenAIObject.construct_from({"content": f"{word} means {defn}", "usage": {"total_tokens": 1}})

    def extract_text_from_response(self, response):
        return response["content"]


def test_completer_get_cloze_text():
    completer = DummyCompleter(model_id="my_dummy_completer")
    result = completer.get_cloze_text("apple", "banana")
    expected = "apple means banana"
    assert result == expected
