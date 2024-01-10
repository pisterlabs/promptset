import openai
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path="config.conf")

openai.api_key = os.getenv("OPENAI_API_KEY")

def get_transcript(audio_file):
    file = open(audio_file, "rb")
    transcript = openai.Audio.transcribe("whisper-1", file=file)
    return transcript["text"]