import pytest
from unittest.mock import patch, Mock
import openai

from src.generate_messages import gpt_commit_message_generation

def mock_openai_response():
    class Choice:
        def __init__(self):
            self.message = self

        content = "Test message content"

    class Response:
        choices = [Choice()]

    return Response()

def test_gpt_commit_message_generation_success(mocker):
    mocker.patch('openai.ChatCompletion.create', return_value=mock_openai_response())

    response = gpt_commit_message_generation()
    assert response == "Test message content", "The response should match the mocked message content"

def test_gpt_commit_message_generation_openai_error(mocker):
    mocker.patch('openai.ChatCompletion.create', side_effect=openai.error.OpenAIError("OpenAI API Error"))

    response = gpt_commit_message_generation()
    assert response == "Default Commit Message", "The response should be the default message on OpenAI API error"

def test_gpt_commit_message_generation_unexpected_error(mocker):
    mocker.patch('openai.ChatCompletion.create', side_effect=Exception("Some unexpected error"))

    response = gpt_commit_message_generation()
    assert response == "Default Commit Message", "The response should be the default message on unexpected error"
