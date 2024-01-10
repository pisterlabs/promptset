from abc import ABCMeta, abstractmethod
import os
import wave
import subprocess
import time
from pathlib import Path
import logging
import sys

import pyaudio
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play

import openai
openai.api_key = os.getenv('OPENAI_API_KEY')

ROOT_PATH = Path('/home/pi')
BASE_PATH = Path('.')


def set_logger(name=None):
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.addFilter(lambda record: record.levelno <= logging.INFO)

    stderr_handler = logging.StreamHandler(stream=sys.stderr)
    stderr_handler.setLevel(logging.WARNING)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(stdout_handler)
    logger.addHandler(stderr_handler)


class TTS:
    def __init__(self):
        self.path_audio = BASE_PATH.joinpath('tts.mp3')

    def tts(self, text: str):
        tts = gTTS(text, lang="en")
        tts.save(self.path_audio)

    def speech(self):
        sound = AudioSegment.from_mp3(self.path_audio)
        play(sound)


class MetaSTT(metaclass=ABCMeta):
    def __init__(self):
        self.path_audio = BASE_PATH.joinpath('tmp.wav')
        self.path_text = BASE_PATH.joinpath('tmp.wav.txt')

    @abstractmethod
    def transcribe(self) -> str:
        pass


class STTWhisperApi(MetaSTT):
    def transcribe(self) -> str:
        file = open(self.path_audio, "rb")
        return openai.Audio.transcribe("whisper-1", file, language='en')['text']


class STTWhisperCpp(MetaSTT):
    def transcribe(self) -> str:
        path_cpp = ROOT_PATH.joinpath('whisper.cpp/main')
        path_whisper = ROOT_PATH.joinpath('whisper.cpp/models/ggml-base.en.bin')

        subprocess.call([path_cpp, 
            '-m', path_whisper,
            '-f', self.path_audio, 
            '-otxt', self.path_text,  # may not work
            ])

        with open(self.path_text, 'r') as f:
            transcript = f.read()
        return transcript
        

class ChatGPT:
    def __init__(self):
        self.initial_prompt = "You're a brilliant English tutor. Let's talk with your student."
        self.messages=[{"role": "system", "content": self.initial_prompt}]
    
    def reply(self, prompt: str) -> str:
        self.messages.append({"role": "user", "content": prompt})
        result = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=self.messages)
        response = result.choices[0].message.content
        self.messages.append({"role": "assistant", "content": response})
        return response


class Recorder:
    def __init__(self):
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.CHUNK = 1024
        self.WAVE_OUTPUT_FILENAME = "tmp.wav"
        self.RECORD_SECONDS = 5

    def record(self):
            self.audio = pyaudio.PyAudio()
            self.stream = self.audio.open(
                format=self.FORMAT, 
                channels=self.CHANNELS,
                rate=self.RATE, 
                input=True, 
                frames_per_buffer=self.CHUNK
                )

            self.frames = []
            for i in range(0, int(self.RATE / self.CHUNK * self.RECORD_SECONDS)):
                self.data = self.stream.read(self.CHUNK)
                self.frames.append(self.data)

            self.stream.stop_stream()
            self.stream.close()
            self.audio.terminate()

            self.waveFile = wave.open(self.WAVE_OUTPUT_FILENAME, 'wb')
            self.waveFile.setnchannels(self.CHANNELS)
            self.waveFile.setsampwidth(self.audio.get_sample_size(self.FORMAT))
            self.waveFile.setframerate(self.RATE)
            self.waveFile.writeframes(b''.join(self.frames))
            self.waveFile.close()

    
if __name__=='__main__':
    set_logger()
    logger = logging.getLogger()

    recorder = Recorder()
    stt = STTWhisperApi()
    chat_gpt = ChatGPT()
    tts = TTS()

    # Initial prompt
    # prompt = "Let's role-play some situation! Can you suggest me 2 situations to learn?"
    # print("# You > ", prompt)
    # response = chat_gpt.reply(prompt)
    # print("# ChatGPT > ", response)
    # tts.tts(response)
    # tts.speech()

    try:
        while True:
            print('# recording (5 sec) ...')
            recorder.record()
            # print("# recorded")

            # print('# transcribing ...')
            transcript = stt.transcribe()
            print("# You > ", transcript)

            # print('# waiting for ChatGPT ...')
            response = chat_gpt.reply(transcript)
            print("# ChatGPT > ", response)
            tts.tts(response)
            tts.speech()

    except KeyboardInterrupt:
        print('\nterminated')
