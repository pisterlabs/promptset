"""Speech router controller."""
import logging
import pathlib
import tempfile

import fastapi
import ffmpeg
from fastapi import status

from linguaweb_api.core import config
from linguaweb_api.microservices import openai

settings = config.get_settings()
LOGGER_NAME = settings.LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)

TARGET_FILE_FORMAT = ".mp3"


async def transcribe(audio: fastapi.UploadFile) -> str:
    """Transcribes audio using OpenAI's Whisper.

    Args:
        audio: The audio file.

    Returns:
        str: The transcription of the audio as a string. The string is
            stripped of newlines and converted to lowercase.
    """
    logger.debug("Transcribing audio.")
    await _check_file_size(audio, max_size=1024 * 1024)

    with tempfile.TemporaryDirectory() as temp_dir:
        target_path = pathlib.Path(temp_dir) / f"audio{TARGET_FILE_FORMAT}"
        _convert_audio(audio, temp_dir, target_path)
        return await openai.SpeechToText().run(target_path)


async def _check_file_size(audio: fastapi.UploadFile, max_size: int) -> None:
    """Check if the size of the audio file exceeds the maximum allowed size.

    Args:
        audio: The audio file to check.
        max_size: The maximum allowed size in bytes.

    Raises:
        fastapi.HTTPException: 413 If the audio file size exceeds the maximum
        allowed size or if the audio file is not seekable.
    """
    fs = await audio.read()
    if len(fs) > max_size:
        msg = f"Audio file size exceeds maximum allowed size of {max_size} bytes."
        logger.error(msg)
        raise fastapi.HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=msg,
        )
    await audio.seek(0)


def _convert_audio(
    audio: fastapi.UploadFile,
    directory: str,
    target_path: pathlib.Path,
) -> None:
    """Converts the audio to the target format.

    Args:
        audio: The audio file.
        directory: The directory to save the audio file to.
        target_path: The path to save the audio file to.

    Raises:
        fastapi.HTTPException: 400 If the audio file does not have a filename.
    """
    if audio.filename is None:
        raise fastapi.HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The audio file must have a filename.",
        )

    extension = pathlib.Path(audio.filename).suffix
    if extension == TARGET_FILE_FORMAT:
        logger.debug("Audio is already in the correct format.")
        with target_path.open("wb") as target_file:
            target_file.write(audio.file.read())
    else:
        logger.debug("Converting audio to correct format.")
        audio_path = pathlib.Path(directory) / f"audio{extension}"
        with audio_path.open("wb") as audio_file:
            audio_file.write(audio.file.read())
        ffmpeg.input(str(audio_path)).output(str(target_path)).run()
