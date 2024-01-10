from typing import Literal

from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()


class SpeechGenerator:
    def __init__(self, api_key: str, model: str = "tts-1", voice="fable"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.voice = voice

    def generate_speech(self, text, file_path):
        response = self.client.audio.speech.create(
            model=self.model,
            voice=self.voice,
            input=text
        )
        response.stream_to_file(file_path)


def main():
    api_key = os.getenv("api_key")
    speech_generator = SpeechGenerator(api_key)

    text = "Hi Everyone, This is Donald Trump. I'm coming back to clean up the mess!"
    speech_file_path = "speech2.mp3"

    speech_generator.generate_speech(text, speech_file_path)
    print("----Success----")


if __name__ == '__main__':
    main()
