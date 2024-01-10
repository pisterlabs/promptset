from typing import List

from openai import AsyncOpenAI
from openai.types.audio import Transcription

from ..models.configuration.base import Configuration


async def exec_whisper(
    conf: Configuration,
    file: str,
    prompt: str = None,
) -> List[Transcription]:

    client = AsyncOpenAI(
        api_key=conf.api_key,
        max_retries=conf.max_retries,
        organization=conf.organization
    )

    response: str = await client.audio.transcriptions.create(
        file=open(file, 'rb'),
        model='whisper-1',
        language=conf.language,
        response_format='text',
        prompt=prompt,
        temperature=conf.temperature,
    )


    return response
