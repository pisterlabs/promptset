import openai
import os
import time
from pathlib import Path
from moviepy.editor import AudioFileClip
import moviepy
import elevenlabs

if __name__ == "__main__":
    import config

client = openai.Client()

class Speech:
    path: Path
    sentence: str
    audio: AudioFileClip

    def __init__(self, sentence: str, path: Path):
        self.sentence = sentence
        self.audio = AudioFileClip(path.__str__())

    def __str__(self):
        return self.sentence

    @property
    def duration(self):
        return self.audio.duration

class SpeechGenOpenAI:
    model = "nova"
    voice = "nova"

    def generate(self, text: str, path: Path):
        resp = client.audio.speech.create(input=text, model=model, voice=voice)
        resp.stream_to_file(path)

class SpeechGen11:
    voice = "Daniel"
    model = "eleven_multilingual_v2"

    def generate(self, text: str, path: Path):
        resp = elevenlabs.generate(
            text = text,
            voice = self.voice,
            model = self.model,
        )
        elevenlabs.save(resp, path)


def speech_from_sentences(sentences: [str], speech_generator=SpeechGen11()) -> [Speech]:
    parent_dir = Path(__file__).parent  / str(int(time.time()))
    os.mkdir(parent_dir)

    ss = []
    for i, sentence in enumerate(sentences):
        speech_file_path = parent_dir / f"{i}.mp3"
        speech_generator.generate(text=sentence, path=speech_file_path)
        ss.append(Speech(sentence, speech_file_path))

    return ss


if __name__ == "__main__":
    speech_from_sentences(["hello, how are you doing today my name is Daniel!"])
