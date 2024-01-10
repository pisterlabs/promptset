from unittest.mock import patch
import pytest
from unittest.mock import AsyncMock
from fastapi.testclient import TestClient
from openai.error import OpenAIError, APIError, Timeout, RateLimitError, APIConnectionError, InvalidRequestError, AuthenticationError, ServiceUnavailableError
from app import app

client = TestClient(app)

class MockResponse:
    async def __aiter__(self):
        yield {
            'choices': [
                {
                    'delta': {
                        'content': 'Hello, world!'
                    }
                }
            ]
        }


@pytest.mark.asyncio
async def test_endpoint():
    with patch('openai.ChatCompletion.acreate', new_callable=AsyncMock) as mock_acreate:
        # Define the mock response
        mock_acreate.return_value = MockResponse()

        # Make request to application
        response = client.post("/gpt4", json={
            'messages': [
                {'role': 'system', 'content': 'Act like an assistant'},
                {'role': 'user', 'content': 'Hello'}
            ],
            'model_type': 'gpt-3.5-turbo'
        })

        # Check the response
        assert response.status_code == 200
        assert response.content.decode() == 'Hello, world!'


# List of error types and messages to test
error_types_and_messages = [
    (APIError, 'Test APIError'),
    (Timeout, 'Test Timeout'),
    (RateLimitError, 'Test RateLimitError'),
    (APIConnectionError, 'Test APIConnectionError'),
    (InvalidRequestError, 'Test InvalidRequestError'),
    (AuthenticationError, 'Test AuthenticationError'),
    (ServiceUnavailableError, 'Test ServiceUnavailableError'),
]

@pytest.mark.parametrize('error_type,error_message', error_types_and_messages)
@pytest.mark.asyncio
async def test_OpenAIError(error_type, error_message):
    # Create the error object
    if error_type is InvalidRequestError:
        error = error_type(message=error_message, param='dummy_param')
    else:
        error = error_type(error_message)

    with patch('openai.ChatCompletion.acreate', new_callable=AsyncMock, side_effect=error) as mock_acreate:
        # Make request to application
        response = client.post("/gpt4", json={
            'messages': [
                {'role': 'system', 'content': 'Act like an assistant'},
                {'role': 'user', 'content': 'Hello'}
            ],
            'model_type': 'gpt-3.5-turbo'
        })

        # Check the response
        assert response.status_code == 200
        assert response.content.decode() == f'{error_type.__name__}: {error_message}'
