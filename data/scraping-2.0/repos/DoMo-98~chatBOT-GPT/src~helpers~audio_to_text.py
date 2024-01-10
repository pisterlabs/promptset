"""This module contains the function for converting audio to text."""

# Standard library imports
from io import BytesIO

# Third party imports
import openai
from pydub import AudioSegment
from telegram import File

# Local application imports
from src.constants import (OPENAI_API_KEY, WHISPER_MODEL)


async def download_audio(audio: File) -> BytesIO:
    """Download the audio file."""
    audio_data = BytesIO()
    audio.download(out=audio_data)
    audio_data.seek(0)
    return audio_data

def convert_audio_to_wav(audio_data: BytesIO) -> BytesIO:
    """Convert the audio file to wav."""
    audio_segment = AudioSegment.from_ogg(audio_data)
    wav_data = BytesIO()
    audio_segment.export(wav_data, format='wav')
    wav_data.seek(0)
    return wav_data

async def transcribe_audio(wav_data: BytesIO) -> str:
    """Transcribe the audio using OpenAI's Whisper."""
    wav_data.name = "temp_audio.wav"

    openai.api_key = OPENAI_API_KEY
    transcript = await openai.Audio.atranscribe(WHISPER_MODEL, wav_data)

    # return transcript['text']
    return transcript.get('text', None)

async def audio_to_text(audio: File) -> str:
    """Convert audio to text."""
    audio_data = await download_audio(audio)
    wav_data = convert_audio_to_wav(audio_data)
    text = await transcribe_audio(wav_data)
    return text
