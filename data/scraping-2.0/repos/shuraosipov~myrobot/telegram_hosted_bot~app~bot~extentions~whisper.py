
from openai import OpenAI

client = OpenAI()

async def transcribe_audio(file_path) -> str:
    """ Transcribe the audio file """
    audio_file= open(file_path, "rb")
    transcript = client.audio.transcribe("whisper-1", audio_file, response_format="text")
    return transcript



