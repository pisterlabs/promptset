from typing import Generator
from unittest.mock import MagicMock

import openai
from openai.openai_object import OpenAIObject

from mentor_mingle import CFGGpt, ChatHandler, Gpt
from mentor_mingle.persona.mentor import Mentor


class TestChatHandler:
    """
    Test the ChatHandler class.
    """

    def test_chat_handler_init(self, fake_redis):
        """
        Test the stream_chat method of the ChatHandler class.
        """
        handler = ChatHandler(cache_client=fake_redis)
        assert isinstance(handler.model, Gpt)
        assert isinstance(handler.agent, Mentor)
        assert isinstance(handler.model.config, CFGGpt)

    def mock_response_generator(self, **kwargs) -> Generator[OpenAIObject, None, None]:
        """
        Mock the response generator, Another update.

        Args:
            None

        Returns:
            Generator[OpenAIObject, None, None]: A generator of OpenAIObjects
        """
        mock_data = [{"choices": [{"delta": {"content": "Hello"}}]}, {"choices": [{"delta": {"content": "Welcome!"}}]}]
        for data in mock_data:
            mock_obj = MagicMock()
            choice_mock = MagicMock()
            choice_mock.delta = data["choices"][0]["delta"]
            mock_obj.choices = [choice_mock]
            yield mock_obj

    def test_stream_chat(self, mocker: MagicMock, fake_redis):
        """
        Test the stream_chat method of the ChatHandler class.

        Args:
            mocker (MagicMock): The mocker object

        Returns:
            None
        """
        # Create a mock instance of your class
        llm = ChatHandler(cache_client=fake_redis)

        # Patch the API call to return mock_response
        mocker.patch.object(openai.ChatCompletion, "create", side_effect=self.mock_response_generator)

        responses = list(llm.stream_chat("Hello, Mentor!"))
        assert responses == ["Hello", "Welcome!"]
