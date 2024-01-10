import warnings
import os
import json
from abc import ABC, abstractmethod
from typing import cast, Any

import whisper
import openai
from dotenv import load_dotenv
# from vosk import Model, KaldiRecognizer
import vosk


from src.core.wav_file_writer import WavFileWriter

load_dotenv(".env.local")
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORG_ID")

wav_file_writer = WavFileWriter()

class Asr(ABC):
    @abstractmethod
    def __call__(self, audio_file_byte) -> None:
        # pass the audio byte into the buffer. For streaming model, load it to the model also
        pass

    @abstractmethod
    def fetch(self) -> str:
        # return the content of the output buffer
        pass

    @abstractmethod
    def reset(self) -> None:
        # reset the output buffer
        pass

class WhisperLocal(Asr):
    def __init__(self, model_size="small"):
        self.model = whisper.load_model(model_size)
        self.audio_byte_buffer = b""
        self.output_buffer = ""

    def __call__(self, audio_bytes):
        self.audio_byte_buffer += audio_bytes

    def fetch(self):
        audio_file_path = "./tmp.wav"
        wav_file_writer(self.audio_byte_buffer, audio_file_path)
        transcribe_text: str = cast(str, self.model.transcribe(audio_file_path)["text"])
        self.output_buffer += transcribe_text
        self._reset_byte_buffer()
        return self.output_buffer

    def reset(self):
        self.audio_byte_buffer = b""
        self.output_buffer = ""

    def _reset_byte_buffer(self):
        self.audio_byte_buffer = b""


class WhisperCloud(Asr):
    def __init__(self, model="whisper-1"):
        self.model = model
        self.audio_byte_buffer = b""
        self.output_buffer = ""

    def __call__(self, audio_bytes):
        self.audio_byte_buffer += audio_bytes

    def fetch(self):
        audio_file_path = "./tmp.wav"
        wav_file_writer(self.audio_byte_buffer, audio_file_path)
        with open(audio_file_path, "rb") as audio_file:
            res = cast(dict[str, Any], openai.Audio.transcribe(self.model, audio_file))
            self.output_buffer += res["text"]
        self._reset_byte_buffer()
        return self.output_buffer

    def reset(self):
        self.audio_byte_buffer = b""
        self.output_buffer = ""

    def _reset_byte_buffer(self):
        self.audio_byte_buffer = b""


class Vosk(Asr):
    def __init__(self, model_path="./models/vosk-model-en-us-0.22"):
        self.model = vosk.Model(model_path)
        self.recognizer = vosk.KaldiRecognizer(self.model, 16000)
        self.output_buffer = ""

    def __call__(self, audio):
        if self.recognizer.AcceptWaveform(audio):
            self.output_buffer = json.loads(self.recognizer.Result())["text"]
        else:
            self.output_buffer =  json.loads(self.recognizer.PartialResult())["partial"]

    def fetch(self):
        return self.output_buffer

    def reset(self):
        self._reset_byte_buffer()
        self.output_buffer = ""

    def _reset_byte_buffer(self):
        self.recognizer = vosk.KaldiRecognizer(self.model, 16000)
