import os
from pathlib import Path

from openai import OpenAI

from videopython.utils.common import generate_random_name


def text_to_speech_openai(text: str, voice: str = "alloy", output_dir: Path | None = None) -> Path:
    client = OpenAI()
    filename = generate_random_name(suffix=".mp3")

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(os.getcwd())

    save_path = output_dir / filename

    response = client.audio.speech.create(model="tts-1", voice=voice, input=text)
    response.stream_to_file(save_path)

    return save_path
