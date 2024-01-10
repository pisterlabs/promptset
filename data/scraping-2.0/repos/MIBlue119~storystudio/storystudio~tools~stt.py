import os

import openai

from storystudio.settings import app_settings
from storystudio.utils import log_io

openai.api_key = app_settings.OPENAI_API_KEY


def generate_srt(audio_path, caption_output_path):
    with open(audio_path, "rb") as audio_file:
        transcript = openai.Audio.transcribe(
            file=audio_file,
            model="whisper-1",
            response_format="srt",
        )
        with open(caption_output_path, "w") as caption_file:
            # Write the transcript to a file
            caption_file.write(transcript)


@log_io
def generate_voice_to_text(voice_output_dir, voice_to_text_output_dir):
    os.makedirs(voice_to_text_output_dir, exist_ok=True)
    for file in os.listdir(voice_output_dir):
        if file.endswith(".wav"):
            audio_path = os.path.join(voice_output_dir, file)
            caption_output_path = os.path.join(
                voice_to_text_output_dir, file.replace(".wav", ".srt")
            )
            generate_srt(audio_path, caption_output_path)
