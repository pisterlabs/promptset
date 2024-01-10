"""Transcribe audio."""
import os
import tempfile
from pathlib import Path
from typing import cast

import openai
from dotenv import load_dotenv
from pydub import AudioSegment

# https://platform.openai.com/docs/api-reference/audio/create
AUDIO_SUFFIXES = [
    "mp3",
    "mp4",
    "mpeg",
    "mpga",
    "m4a",
    "wav",
    "webm",
]


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def transcribe_audio_file(audio_file_path: Path) -> str:
    """Transcribe audio file using OpenAI Audio API.

    https://platform.openai.com/docs/api-reference/audio/create
    """
    if audio_file_path.suffix in AUDIO_SUFFIXES:
        msg = "File type not supported."
        raise ValueError(msg)
    with audio_file_path.open("rb") as f:
        response = openai.Audio.transcribe(  # type: ignore reportUnknownMemberType # noqa: E501
            "whisper-1",
            file=f,
            language="ja",
            response_format="verbose_json",
        )
        text = cast(str, response.text)  # type: ignore reportUnknownMemberType

    return text


def transcribe_audio(audio: AudioSegment) -> str:
    """Transcribe audio."""
    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        audio.export(f.name, format="wav")
        text = transcribe_audio_file(Path(f.name))

    return text
