import io
import os
from enum import Enum
from typing import Protocol

import openai
import speech_recognition as sr
from pydub import AudioSegment
from pydub.playback import play


class InterfaceType(Enum):
    TERMINAL = "terminal"
    AUDIO = "audio"


class Interface(Protocol):
    def input(self, prefix: str = "") -> str:
        return prefix + self._input_impl()

    def _input_impl(self) -> str:
        ...

    def output(self, message: str) -> None:
        ...


class Terminal(Interface):
    def __init__(self):
        pass

    def _input_impl(self) -> str:
        return input("Please input your command. > ")

    def output(self, message: str) -> None:
        print("Robot: {}".format(message))


class Audio(Interface):
    def __init__(self):
        self.r = sr.Recognizer()
        self.mic = sr.Microphone()
        self.client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def _input_impl(self) -> str:
        print("Please tell me your command.")
        with self.mic as source:
            self.r.adjust_for_ambient_noise(source)
            audio = self.r.listen(source)

        try:
            return self.r.recognize_whisper(audio, language="japanese")

        except sr.UnknownValueError:
            print("could not understand audio")
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))

    def output(self, message: str) -> None:
        response = self.client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=message,
        )
        byte_stream = io.BytesIO(response.content)
        audio = AudioSegment.from_file(byte_stream, format="mp3")
        play(audio)
