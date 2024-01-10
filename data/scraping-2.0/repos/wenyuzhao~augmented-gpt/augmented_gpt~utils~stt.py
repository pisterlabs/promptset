from typing import Literal

import openai
from pathlib import Path


class SpeechToText:
    def __init__(self, api_key: str, model: Literal["whisper-1"] = "whisper-1"):
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model = model

    async def transcribe(self, path: str | Path) -> str:
        with open(path, "rb") as audio_file:
            transcript = await self.client.audio.transcriptions.create(
                model="whisper-1", file=audio_file
            )
            return transcript.text

    def transcribe_sync(self, path: str | Path) -> str:
        from . import block_on

        return block_on(self.transcribe(path))


__all__ = ["SpeechToText"]
