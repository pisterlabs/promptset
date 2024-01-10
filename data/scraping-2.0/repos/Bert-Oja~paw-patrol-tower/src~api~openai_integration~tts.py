"""
This module contains the functions that interact with the OpenAI Text-to-Speech (TTS) model.
"""
import os
import logging
from typing import Optional
import openai


def create_mission_audio(path: str, text: str) -> Optional[bool]:
    """
    Creates an audio file from the given text using the OpenAI Text-to-Speech (TTS) model.

    Args:
        path (str): The path where the audio file will be saved.
        text (str): The text to be converted into speech.

    Returns:
        Optional[bool]: True if the audio file was successfully created and saved, False otherwise.
    """
    try:
        response = openai.audio.speech.create(
            model="tts-1", voice=os.getenv("TTS_VOICE", "nova"), input=text
        )
        response.stream_to_file(path)
        return True
    except (
        openai.APITimeoutError,
        openai.APIConnectionError,
        openai.BadRequestError,
        openai.AuthenticationError,
        PermissionError,
        openai.RateLimitError,
        openai.APIError,
    ) as e:
        logging.error("OpenAI API request failed: %s", e)
        return False
