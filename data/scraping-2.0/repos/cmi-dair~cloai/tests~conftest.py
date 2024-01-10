"""Test configurations."""
import os
from unittest import mock

import pytest
import pytest_mock

from cloai import openai_api


def pytest_configure() -> None:
    """Configure pytest with the necessary environment variables.

    Args:
        config: The pytest configuration object.

    """
    os.environ["OPENAI_API_KEY"] = "API_KEY"


@pytest.fixture()
def mock_openai(mocker: pytest_mock.MockFixture) -> mock.MagicMock:
    """Mocks the OpenAI client."""
    mock_speech_create = mocker.MagicMock()
    mock_transcriptions_create = mocker.MagicMock()
    mock_audio_speech = mocker.MagicMock(
        speech=mocker.MagicMock(create=mock_speech_create),
        transcriptions=mocker.MagicMock(create=mock_transcriptions_create),
    )
    mock_audio = mocker.MagicMock(audio=mock_audio_speech)
    mock_images = mocker.MagicMock(generate=mocker.MagicMock())
    mock_chat = mocker.MagicMock(
        completions=mocker.MagicMock(create=mocker.MagicMock()),
    )
    mock_client = mocker.MagicMock(
        spec=openai_api.openai.OpenAI,
        audio=mock_audio,
        images=mock_images,
        chat=mock_chat,
    )
    return mocker.patch(
        "cloai.openai_api.openai.OpenAI",
        return_value=mock_client,
    )
