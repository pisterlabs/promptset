import os
import io
import logging
from unittest.mock import patch, mock_open, ANY
import pytest


from openai import OpenAIError
from openai.error import Timeout

from transcriber.audio_transcription import (
    _initialize_openai_api_and_logging,
    _transcribe_audio_from_file,
    transcribe_audio
)

mock_transcript = {'text': 'sample text'}


@pytest.fixture
def setup_logging():
    logging.getLogger().setLevel(logging.INFO)

# Test initialize_openai_api_and_logging function
def test_initialize_openai_api_and_logging_missing_api_key():
    with patch('sys.exit') as mock_exit, patch('logging.error') as mock_error:
        os.environ['OPENAI_API_KEY'] = ''
        _initialize_openai_api_and_logging()
        mock_error.assert_called_with('Error - OPENAI_API_KEY not set')
        mock_exit.assert_called_with(1)


@pytest.mark.usefixtures("setup_logging")
def test_initialize_openai_api_and_logging_successful_initialization():
    with patch('dotenv.load_dotenv'):
        os.environ['OPENAI_API_KEY'] = 'fake-key'
        _initialize_openai_api_and_logging()
        assert logging.getLogger().getEffectiveLevel() == logging.INFO
        assert os.getenv('OPENAI_API_KEY') == 'fake-key'


def test_transcribe_audio_from_file_successful():
    # Mock the openai.Audio.transcribe function
    with patch('openai.Audio.transcribe', return_value=mock_transcript):
        fake_audio_data = b'fake audio data'
        fake_audio_file_obj = io.BytesIO(fake_audio_data)
        result = _transcribe_audio_from_file(fake_audio_file_obj)
        assert result == 'sample text'


def test_transcribe_audio_file_not_found():
    with patch('logging.error') as mock_log_error:
        with pytest.raises(FileNotFoundError):
            transcribe_audio('non_existent_file')
    mock_log_error.assert_called_with(
        'Error accessing audio file %s: %s', 'non_existent_file', ANY
        )


def test_transcribe_audio_openai_error():
    with patch('openai.Audio.transcribe', side_effect=OpenAIError('OpenAI Error')), \
         patch('logging.error') as mock_log_error:
        # Create a BytesIO object with some fake audio data
        fake_audio_file_obj = io.BytesIO(b'fake audio data')
        with pytest.raises(OpenAIError, match='OpenAI Error'):
            _transcribe_audio_from_file(fake_audio_file_obj)

    mock_log_error.assert_called_with(
        'Error calling OpenAI API: %s', ANY
        )


def test_transcribe_audio_successful():
    fake_transcription = "This is a fake transcription."
    with patch('openai.Audio.transcribe', return_value={"text": fake_transcription}):
        fake_audio_file_obj = io.BytesIO(b'fake audio data')
        result = _transcribe_audio_from_file(fake_audio_file_obj)

    assert result == fake_transcription


def test_transcribe_audio_retry_on_openai_error():
    m_open = mock_open(read_data="fake audio data")

    expected_exception = Timeout('Retry')

    with patch('builtins.open', m_open), \
         patch('openai.Audio.transcribe', side_effect=[expected_exception, mock_transcript]), \
         patch('logging.error') as mock_log_error, \
         patch('logging.info') as mock_log_info:

        transcribe_audio('fake_audio_file')

        mock_log_error.assert_called_with('Error calling OpenAI API: %s', expected_exception)
        mock_log_info.assert_called_with(mock_transcript['text'])
