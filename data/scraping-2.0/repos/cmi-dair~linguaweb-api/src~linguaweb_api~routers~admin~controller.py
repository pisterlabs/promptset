"""Controller for the listening router."""
import asyncio
import logging
from typing import NamedTuple

import fastapi
from fastapi import status
from sqlalchemy import orm

from linguaweb_api.core import config, dictionary, models
from linguaweb_api.microservices import openai, openai_constants, s3

settings = config.get_settings()
LOGGER_NAME = settings.LOGGER_NAME
OPENAI_VOICE = settings.OPENAI_VOICE
logger = logging.getLogger(LOGGER_NAME)


async def add_word(word: str, session: orm.Session, s3_client: s3.S3) -> models.Word:
    """Adds a word to the database.

    Args:
        word: The word to add.
        session: The database session.
        s3_client: The S3 client to use.

    Returns:
        The word model.
    """
    logger.debug("Adding word.")
    word_model = session.query(models.Word).filter_by(word=word).first()
    if word_model:
        raise fastapi.HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Word already exists in database.",
        )

    logger.debug("Word does not exist in database.")
    text_tasks_promise = _get_text_tasks(word)
    listening_bytes_promise = _get_listening_task(word)
    s3_key = f"{word}_{OPENAI_VOICE.value}.mp3"

    text_tasks, listening_bytes = await asyncio.gather(
        text_tasks_promise,
        listening_bytes_promise,
    )

    logger.debug("Creating new word.")
    new_word = models.Word(
        word=word,
        description=text_tasks.description,
        synonyms=text_tasks.synonyms,
        antonyms=text_tasks.antonyms,
        jeopardy=text_tasks.jeopardy,
        s3_key=s3_key,
    )
    s3_client.create(key=s3_key, data=listening_bytes)
    session.add(new_word)
    session.commit()
    logger.debug("Added word.")
    return new_word


async def add_preset_words(session: orm.Session, s3_client: s3.S3) -> list[models.Word]:
    """Adds preset words to the database.

    Args:
        session: The database session.
        s3_client: The S3 client to use.

    Returns:
        The word models.
    """
    logger.debug("Adding preset words.")
    preset_words = dictionary.read_words()
    preset_models = (
        session.query(models.Word).filter(models.Word.word.in_(preset_words)).all()
    )
    if len(preset_models) == len(preset_words):
        raise fastapi.HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="All preset words already exist in database.",
        )

    logger.debug("Added preset words.")
    return await asyncio.gather(
        *[
            add_word(word, session, s3_client)
            for word in preset_words
            if not session.query(models.Word).filter_by(word=word).first()
        ],
    )


class _TextTasks(NamedTuple):
    """Named tuple for the text tasks."""

    description: str
    synonyms: str
    antonyms: str
    jeopardy: str


async def _get_text_tasks(word: str) -> _TextTasks:
    """Runs GPT to get text tasks.

    Args:
        word: The word to get text tasks for.

    Returns:
        The text tasks.
    """
    logger.debug("Running GPT.")
    gpt = openai.GPT()
    gpt_calls = [
        gpt.run(prompt=word, system_prompt=openai_constants.Prompts.WORD_DESCRIPTION),
        gpt.run(prompt=word, system_prompt=openai_constants.Prompts.WORD_SYNONYMS),
        gpt.run(prompt=word, system_prompt=openai_constants.Prompts.WORD_ANTONYMS),
        gpt.run(prompt=word, system_prompt=openai_constants.Prompts.WORD_JEOPARDY),
    ]

    results = await asyncio.gather(*gpt_calls)
    return _TextTasks(*results)


async def _get_listening_task(word: str) -> bytes:
    """Fetches audio for a given word from OpenAI.

    Args:
        word: The word to fetch audio for.

    Returns:
        The audio bytes and the S3 key.
    """
    tts = openai.TextToSpeech()
    return await tts.run(word)
