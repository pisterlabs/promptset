import os
import sys
import logging
from typing import BinaryIO, Optional

import openai

from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type
from dotenv import load_dotenv


AUDIO_ENGINE_ID = "whisper-1"


retry_conditions = (
    retry_if_exception_type(openai.error.Timeout)
    | retry_if_exception_type(openai.error.APIError)
    | retry_if_exception_type(openai.error.APIConnectionError)
    | retry_if_exception_type(openai.error.RateLimitError)
    | retry_if_exception_type(openai.error.ServiceUnavailableError)
)

configured_retry = retry(
    wait=wait_random_exponential(multiplier=0.5, max=60),
    stop=stop_after_attempt(3),
    reraise=True,
    retry=retry_conditions,
)


def _initialize_openai_api_and_logging() -> None:
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        logging.error("Error - OPENAI_API_KEY not set")
        sys.exit(1)
    else:
        openai.api_key = openai_api_key

    logging.basicConfig(level=logging.INFO)


def _transcribe_audio_from_file(file_obj: BinaryIO) -> Optional[str]:
    """Transcribes audio from a file object using OpenAI Whisper API.

    Args:
        file_obj: A binary file object containing audio data.

    Returns:
        The transcription text if successful, else None.
    """
    try:
        logging.info("Transcribing audio...")
        transcript = openai.Audio.transcribe(AUDIO_ENGINE_ID, file_obj)
        logging.info(transcript["text"])
        return transcript["text"]
    except openai.error.OpenAIError as e:  # pylint: disable=invalid-name
        logging.error("Error calling OpenAI API: %s", e)
        raise


_transcribe_audio_from_file_with_retry = configured_retry(_transcribe_audio_from_file)


def transcribe_audio(file_name: str) -> Optional[str]:
    """Transcribes audio from a file using its name.

    Args:
        file_name: Name of the file containing audio data.

    Returns:
        The transcription text if successful, else None.
    """
    try:
        with open(file_name, "rb") as audio_file_obj:
            logging.info("Opened audio file: %s", file_name)
            return _transcribe_audio_from_file_with_retry(audio_file_obj)
    except (FileNotFoundError, PermissionError) as e:  # pylint: disable=invalid-name
        logging.error("Error accessing audio file %s: %s", file_name, e)
        raise


if __name__ == "__main__":
    _initialize_openai_api_and_logging()

    if len(sys.argv) < 2:
        logging.error("Usage: python transcribe.py <audio_file>")
        sys.exit(1)

    audio_file = sys.argv[1]
    transcribe_audio(audio_file)
