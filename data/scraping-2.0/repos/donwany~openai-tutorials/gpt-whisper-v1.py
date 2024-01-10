from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()


class AudioTranscriber:
    def __init__(self, api_key: str, model: str = "whisper-1"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def transcribe_audio_file(self, file_path):
        """transcribe audio file to text"""
        with open(file_path, "rb") as audio_file:
            transcript = self.client.audio.transcriptions.create(
                model=self.model,
                file=audio_file
            )
        return transcript.text


def main():
    api_key = os.getenv("api_key")
    transcriber = AudioTranscriber(api_key)

    audio_file_path = "speech.mp3"
    transcript_text = transcriber.transcribe_audio_file(audio_file_path)

    print(transcript_text)


if __name__ == '__main__':
    main()
