import random
import re
import tempfile
from pathlib import Path

import httpx
import requests
from django.conf import settings
from openai import OpenAI

from core.models import TranscriptionLogs


def transcribe_rps_from_url(url):
    client = OpenAI(
        api_key=random.choice(settings.OPENAI_KEYS),
        timeout=httpx.Timeout(10.0, connect=5.0),
        max_retries=3,
    )

    response = requests.get(url)

    with tempfile.TemporaryDirectory() as tmpdirname:
        audio_file_path = Path(tmpdirname) / "audio.wav"

        with open(audio_file_path, "wb") as fp:
            fp.write(response.content)

        with open(audio_file_path, "rb") as fp:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=fp,
                prompt="ROCK, PAPER, SCISSORS",
            )

    TranscriptionLogs.objects.create(transcript=transcript.text)
    return re.sub(r"\W+", "", transcript.text.lower().strip())
