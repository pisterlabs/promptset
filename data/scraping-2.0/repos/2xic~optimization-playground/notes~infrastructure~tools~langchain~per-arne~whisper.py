"""
Per Arne needs to hear, and we use open-ai whisper model for that :)
"""
from dotenv import load_dotenv
load_dotenv()

import os
import cache
import openai

class Whisper:
    def __init__(self) -> None:
        self.root = os.path.dirname(
            os.path.abspath(__file__)
        )

    def get_transcript(self, name):
        path = os.path.join(
            self.root,
            "record.wav"
        )
        audio_file = open(path, "rb")
        with cache.Cache(audio_file) as ref:
            if ref.cached:
                print("Cache :=)")
                return ref.cached
            else:
                transcript = openai.Audio.transcribe("whisper-1", audio_file)
                return ref.cache(
                    transcript["text"]
                )

if __name__ == "__main__":
    print(
        Whisper().get_transcript("record.wav")
    )
