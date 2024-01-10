import os
from pathlib import Path

import dotenv
import openai

dotenv.load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


class Transcriber:
    """Transcribes the audio file using the OpenAI API
    The reason why argument is list of audio path is that, because of the limit of the API, the audio should be chunked.
    """

    def __init__(
        self,
        list_audio_path: list[Path],
        model_name: str = "whisper-1",
    ) -> None:
        self.model_name = model_name
        self.list_audio_path = list_audio_path

    @staticmethod
    def transcribe(audio_file_path: Path, model_name: str) -> str:
        """Transcribes the audio file using the OpenAI API"""
        with audio_file_path.open("rb") as audio_file:
            transcript = openai.Audio.transcribe(model_name, audio_file, verbose=True)
        return transcript["text"]

    def get_full_transcript(self) -> str:
        """Transcribes all the audio chunks in the given directory and returns the full transcript"""
        transcript_full = ""
        for audio_path in self.list_audio_path:
            if audio_path.is_file() and audio_path.suffix == ".mp3":
                print(f"Transcribing {audio_path.name}...")
                transcript = Transcriber.transcribe(audio_path, self.model_name)
                transcript_full += transcript
        return transcript_full
