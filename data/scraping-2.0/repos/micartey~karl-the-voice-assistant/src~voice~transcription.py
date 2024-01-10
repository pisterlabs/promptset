from loguru import logger

from src.config import openai


def transcripe_audio_file(file: str) -> str:
    """
    Transcripe a text file
    :param file: wav file
    :return: text
    """

    transcription = openai.audio.transcriptions.create(
        model="whisper-1", file=open(file, "rb")
    ).text

    logger.debug(transcription)

    return transcription
