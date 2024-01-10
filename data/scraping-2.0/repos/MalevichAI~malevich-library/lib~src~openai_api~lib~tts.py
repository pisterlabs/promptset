from typing import List

from openai import AsyncOpenAI

from ..models.configuration.base import Configuration


async def exec_tts(
    inputs: str,
    conf: Configuration,
    file: str,
) -> List[str]:

    client = AsyncOpenAI(
        api_key=conf.api_key,
        max_retries=conf.max_retries,
        organization=conf.organization
    )

    response = await client.audio.speech.create(
        input=inputs,
        model=conf.model or 'tts-1',
        voice=conf.voice,
        response_format='mp3',
        speed=conf.speed
    )

    response.stream_to_file(file)

    return file
