import os
import dotenv
from openai import OpenAI
from pathlib import Path
from pydub import AudioSegment
from playsound import playsound
import argparse
import wave
import sounddevice as sd
import numpy as np

class Translater():
    def __init__(self):
        dotenv.load_dotenv()
        self.client = OpenAI()

    def record(self, input_path, duration=5, sample_rate=44100):
        print("Gravando...")

        recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=2, dtype=np.int16)
        sd.wait()

        with wave.open(input_path, 'wb') as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(recording.tobytes())

        print("Gravação finalizada!")

    def transcribe_translate(self, input_path):
        with open(input_path, "rb") as audio:
            translation = self.client.audio.translations.create(
                model="whisper-1",
                file=audio,
                response_format="text"
            )
        return translation
        
    def text_to_speech(self, text):
        output_path = Path(__file__).parent / "output.mp3"
        response = self.client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text
        )

        response.stream_to_file(output_path)

def __main__():
    translater = Translater()

    parser = argparse.ArgumentParser(description="Transcreve, traduz e depois fala o texto de um arquivo de áudio usando Whisper e tts")
    parser.add_argument("--file", type=str, help="Caminho para o arquivo de áudio a ser traduzido.")

    args = parser.parse_args()
    file = args.file

    if file:
        translation = translater.transcribe_translate(file)
    else:
        translater.record("input.mp3")
        translation = translater.transcribe_translate("input.mp3")

    translater.text_to_speech(translation)

    playsound("output.mp3")

if __name__ == "__main__":
    __main__()