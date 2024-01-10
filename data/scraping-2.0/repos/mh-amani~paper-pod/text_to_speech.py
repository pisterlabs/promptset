from openai import OpenAI
from pathlib import Path


def text_to_speech(text):
    client = OpenAI()

    speech_file_path = Path(__file__).parent / "speech.mp3"
    response = client.audio.speech.create(
      model="tts-1-hd",
      voice="onyx",
      input=text, 
      speed=1.1,
    )
    response.stream_to_file(speech_file_path)
    return speech_file_path
