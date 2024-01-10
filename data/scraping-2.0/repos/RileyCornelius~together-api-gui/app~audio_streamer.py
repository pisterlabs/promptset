import os
from queue import Queue
import shutil
import subprocess
import threading
from typing import Iterator, Literal

from dotenv import load_dotenv
import openai
import speech_recognition as sr


class AudioStreamer:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        self.openai = openai.OpenAI(api_key=api_key)
        self.microphone = sr.Microphone()
        self.recognizer = sr.Recognizer()

        self.is_streaming = False
        self.audio = Queue()
        self.text = Queue()

    def start_streaming(self, stream=None):
        self.is_streaming = True
        self.audio = Queue()
        self.text = Queue()
        threading.Thread(target=self._tts_thread, daemon=True).start()
        threading.Thread(target=self._audio_thread, daemon=True).start()

        if stream:
            for chunk in stream:
                print(chunk, end="", flush=True)
                self.text.put(chunk)

    def stop_streaming(self):
        self.is_streaming = False

    def _tts_thread(self):
        sentence = ""
        while self.is_streaming:
            chunk = self.text.get()
            sentence += chunk
            # TODO: add a better way to detect end of sentence
            if chunk and chunk[-1] in ".!?":
                audio_stream = self.text_to_speech_streaming(sentence)
                self.audio.put(audio_stream)
                sentence = ""

    def _audio_thread(self):
        while self.is_streaming:
            self.audio_streaming(self._stream_audio_generator())

    def _stream_audio_generator(self) -> Iterator[bytes]:
        while self.is_streaming:
            sentence_audio = self.audio.get()
            for bytes in sentence_audio.iter_bytes():
                yield bytes

    def text_to_speech_streaming(
        self,
        text: str,
        voice: Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"] = "echo",
        model: Literal["tts-1", "tts-1-hd"] = "tts-1",
        speed: float = 1.0,
    ):
        stream = self.openai.audio.speech.create(
            input=text,
            model=model,
            voice=voice,
            response_format="mp3",
            speed=speed,
            stream=True,
        )
        return stream

    def speech_to_text_whisper(self, audio_file: str):
        try:
            audio_file = open(audio_file, "rb")
            text = self.openai.audio.transcriptions.create(file=audio_file, model="whisper-1", response_format="text")
            return text
        except Exception as error:
            print(f"Speech to text error: {error}")
            return ""

    def audio_streaming(self, audio_stream: Iterator[bytes]) -> bytes:
        if shutil.which("mpv") is None:
            message = (
                "mpv not found, necessary to stream audio. "
                "On mac you can install it with 'brew install mpv'. "
                "On linux and windows you can install it from https://mpv.io/"
            )
            raise ValueError(message)

        mpv_command = ["mpv", "--no-cache", "--no-terminal", "--", "fd://0"]
        mpv_process = subprocess.Popen(
            mpv_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        audio = bytes()
        for chunk in audio_stream:
            if not self.is_streaming:
                mpv_process.terminate()
                break
            if chunk is not None:
                mpv_process.stdin.write(chunk)
                mpv_process.stdin.flush()
                audio += chunk

        if mpv_process.stdin:
            mpv_process.stdin.close()
        mpv_process.wait()

        self.stop_streaming()
        return audio

    def listening(self):
        try:
            with sr.Microphone() as microphone:
                audio = sr.Recognizer().listen(microphone)
                audio_path = self._save_audio(audio.get_wav_data(), "cache")
            return audio_path
        except sr.UnknownValueError:
            print("Error: Could not understand audio")
            return ""

    def _save_audio(self, data: bytes, file_name: str):
        AUDIO_SAVED_DIRECTORY = "audio/"
        file_name = f"{file_name}.wav"
        os.makedirs(AUDIO_SAVED_DIRECTORY, exist_ok=True)
        path = os.path.join(AUDIO_SAVED_DIRECTORY, file_name)
        with open(path, "wb") as f:
            f.write(data)
        return path
