import numpy as np
import speech_recognition as sr
import io
from scipy.io.wavfile import write
import speech_recognition
import openai
import os

from constants import *
from DataManagment import file_system as fs


class SpeechRecognitionController:
    """
    Transcribes audio to text. Current version is using speech_recognition library.
    """

    def __init__(self):
        print("Initializing speech recognition controller...")
        self.recognizer = sr.Recognizer()
        openai.api_key = os.getenv("OPENAI_API_KEY")
        print("Speech recognition controller initialized!")

    def transcribe(self, audio: np.array, sample_rate=44100, precise: bool = False) -> str:
        byte_io = io.BytesIO(bytes())
        write(byte_io, sample_rate, audio)
        result_bytes = byte_io.read()
        audio_data = sr.AudioData(result_bytes, sample_rate, 2)

        try:
            return self.recognizer.recognize_google(audio_data=audio_data, language="en-US")
        except speech_recognition.UnknownValueError:
            return ""

    def transcribe_path(self, path: Path, precise: bool = False) -> str:
        if precise:
            audio_data = open(path.as_posix(), "rb")
            return openai.Audio.transcribe("whisper-1", audio_data)
        else:
            with sr.AudioFile(path.as_posix()) as source:
                audio = self.recognizer.record(source)
            try:
                return self.recognizer.recognize_google(audio_data=audio, language="en-US")
            except speech_recognition.UnknownValueError:
                return ""