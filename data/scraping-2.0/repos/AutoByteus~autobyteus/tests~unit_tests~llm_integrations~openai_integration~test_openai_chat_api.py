import pytest
from unittest.mock import patch
from autobyteus.llm_integrations.openai_integration.openai_chat_api import OpenAIChatApi
from autobyteus.llm_integrations.openai_integration.openai_message_types import AssistantMessage, UserMessage, SystemMessage

@pytest.fixture
def mock_openai_response(monkeypatch):
    """Mock the OpenAI Chat API response."""
    
    def mock_create(*args, **kwargs):
        return {
            'choices': [
                {'message': {'content': 'This is a mock response', 'role': 'assistant'}}
            ]
        }

    monkeypatch.setattr("openai.ChatCompletion.create", mock_create)

def test_process_input_messages_returns_expected_response(mock_openai_response):
    """Ensure the process_input_messages method returns the expected mock response."""
    api = OpenAIChatApi()
    messages = [UserMessage("Hello, OpenAI!")]
    response = api.process_input_messages(messages)
    assert isinstance(response, AssistantMessage)
    assert response.content == "This is a mock response"

def test_process_input_messages_with_empty_messages(mock_openai_response):
    """Ensure the process_input_messages method with an empty list of messages returns the mock response."""
    api = OpenAIChatApi()
    messages = []
    response = api.process_input_messages(messages)
    assert isinstance(response, AssistantMessage)
    assert response.content == "This is a mock response"

def test_system_message_is_always_first(mock_openai_response):
    """Ensure the system message is always the first message."""
    api = OpenAIChatApi()
    messages = [UserMessage("Hello, OpenAI!")]  # Convert to UserMessage instance

    # Using a mock to capture the arguments passed to openai.ChatCompletion.create
    with patch("autobyteus.llm_integrations.openai_integration.openai_chat_api.openai.ChatCompletion.create", autospec=True) as mock_create:
        mock_create.return_value = {
            'choices': [
                {'message': {'content': 'This is a mock response', 'role': 'assistant'}}
            ]
        }
        api.process_input_messages(messages)
        _, kwargs = mock_create.call_args

        # Convert the dictionary message back to UserMessage instance for checking
        first_message = kwargs['messages'][0]

        # Assert that the first message has the role 'system'
        assert first_message['role'] == 'system', "The first message is not a system message."

def test_extract_response_message_valid_response():
    """Test the _extract_response_message method with a valid response."""
    api = OpenAIChatApi()
    
    mock_response = {
        'choices': [
            {'message': {'content': 'This is a valid response', 'role': 'assistant'}}
        ]
    }
    message = api._extract_response_message(mock_response)
    assert isinstance(message, AssistantMessage)
    assert message.content == "This is a valid response"

def test_extract_response_message_invalid_role():
    """Test the _extract_response_message method with an invalid role."""
    api = OpenAIChatApi()
    
    mock_response = {
        'choices': [
            {'message': {'content': 'This is a response with invalid role', 'role': 'user'}}
        ]
    }
    with pytest.raises(ValueError, match=r"Unexpected role in OpenAI API response: user"):
        api._extract_response_message(mock_response)

def test_extract_response_message_missing_role():
    """Test the _extract_response_message method with missing role key."""
    api = OpenAIChatApi()
    
    mock_response = {
        'choices': [
            {'message': {'content': 'This is a response with missing role'}}
        ]
    }
    with pytest.raises(ValueError, match=r"Unexpected structure in OpenAI API response."):
        api._extract_response_message(mock_response)
