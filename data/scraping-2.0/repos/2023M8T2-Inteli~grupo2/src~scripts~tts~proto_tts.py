from pathlib import Path
from openai import OpenAI
import os

client = OpenAI()

client = OpenAI(
  api_key=os.environ.get("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"),)


speech_file_path = Path(__file__).parent / "speech.mp3"
response = client.audio.speech.create(
  model="tts-1",
  voice="alloy",
  input="Today is a wonderful day to build something people love!"
)

response.stream_to_file(speech_file_path)