import openai
import io
from dotenv import dotenv_values

config = dotenv_values(".env")

OPENAI_KEY = config["OPENAI_KEY"]

openai.api_key = OPENAI_KEY


def transcribe_audio(audio_buffer: io.BytesIO):
    transcript = openai.Audio.transcribe("whisper-1", audio_buffer)
    return transcript["text"]
