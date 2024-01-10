"""
AUDIO TRANSCRIBER

This code transcribes an audio file into text using OpenAI's API.
OpenAI documentation: https://platform.openai.com/docs/guides/speech-to-text/quickstart

The audio snippet used is from chapter two of the book "Umbrellas and Their History" by William Sangster, published in 1855.
Recording from LibriVox: https://librivox.org/umbrellas-and-their-history-by-william-sangster/.

"""

from pathlib import Path
import os
import openai

from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

with open(Path.cwd() / "resources/umbrellasandtheirhistory_snippet.mp4", "rb") as audio_file: 
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    print(transcript)
