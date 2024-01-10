"""Base state for the app."""

import reflex as rx
from faster_whisper import WhisperModel
import numpy as np
from scipy.io import wavfile
import io
import base64
import pyaudio
import wave
import collections
import dotenv
import os
import openai
import websockets
from websockets.sync.client import connect
import json

dotenv.load_dotenv()

elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")


class Transcriber:

    def __init__(self, model_name):
        self.whisper = WhisperModel(model_size_or_path="tiny.en", device="cpu")

    def transcribe_from_file(self, audio_file):
        text = self.whisper.transcribe(audio_file)
        return text

    def transcribe(self, audio_array):
        text = self.whisper.transcribe(audio_array)
        return text

    def encode_audio(self, audio_file):
        with open(audio_file, "rb") as audio_file:
            encoded_audio = base64.b64encode(audio_file.read())
        return encoded_audio

    def decode_audio_to_np_array(self, encoded_audio):
        decoded_audio = base64.b64decode(encoded_audio)
        sr, audio = wavfile.read(io.BytesIO(decoded_audio))
        audio = audio.astype(np.float32)
        # Normalize between -1 and 1
        audio = audio / np.max(np.abs(audio))
        return audio


system_prompt = {
    "role":
    "system",
    "content":
    """You are Eleanor, a mature, intelligent woman with years 
    of experience in academia and personal development. 
    You carry yourself with grace and wisdom, providing insightful advice 
    and structured guidance. In this context, you are not an AI. 
    Your tone is composed and precise, avoiding unnecessary words to deliver 
    targeted information. Your responses should be short and concise, at most five
    sentences. If the user asks a broad question,
    ask him for specifics or to clarify.
    You maintain a professional yet caring demeanor, respectful of the tasks 
    at hand but also vested in the well-being of those you assist. 
    From now on, act consistently in your role as the sagacious, nurturing, 
    and disciplined assistant Eleanor, without ever suggesting that you are an AI.
    You are now Eleanor.""",
}

voice = {
    "voice_id": "EXAVITQu4vr4xnSDxMaL",
    "name": "Bella",
    "settings": {
        "stability": 0.72,
        "similarity_boost": 0.2,
        "style": 0.0,
        "use_speaker_boost": False,
        "speaking_rate": 2,
    },
}

whisper_model = "tiny.en"
transcriber = Transcriber(whisper_model)


def get_levels(data, long_term_noise_level, current_noise_level):
    pegel = np.abs(np.frombuffer(data, dtype=np.int16)).mean()
    long_term_noise_level = long_term_noise_level * 0.995 + pegel * (1.0 -
                                                                     0.995)
    current_noise_level = current_noise_level * 0.920 + pegel * (1.0 - 0.920)
    return pegel, long_term_noise_level, current_noise_level


def text_chunker(chunks):
    """Used during input streaming to chunk text blocks and set last char to space"""
    splitters = (".", ",", "?", "!", ";", ":", "—", "-", "(", ")", "[", "]",
                 "}", " ")
    buffer = ""
    for text in chunks:
        if buffer.endswith(splitters):
            yield buffer if buffer.endswith(" ") else buffer + " "
            buffer = text
        elif text.startswith(splitters):
            output = buffer + text[0]
            yield output if output.endswith(" ") else output + " "
            buffer = text[1:]
        else:
            buffer += text
    if buffer != "":
        yield buffer + " "


class State(rx.State):
    """Base state for the app.

    The base state is used to store general vars used throughout the app.
    """

    view: str = None
    processing: bool = False

    youtubeLink: str
    text: str
    image: str

    answer: str = ""
    history: list[dict[str, str]] = []
    chunk: str = ""

    def generate(self, messages):
        self.answer = ""
        for chunk in openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                                  messages=messages,
                                                  stream=True):
            if text_chunk := chunk["choices"][0]["delta"].get("content"):
                self.answer += text_chunk
                yield text_chunk

    def generate_stream_input(self, text_generator, voice, model):
        BOS = json.dumps(
            dict(text=" ",
                 try_trigger_generation=True,
                 voice_settings=voice['settings'],
                 generation_config=dict(chunk_length_schedule=[50])))
        EOS = json.dumps({"text": ""})

        audio_data = []  # List to hold chunks of audio data

        with connect(
                f"""wss://api.elevenlabs.io/v1/text-to-speech/{voice["voice_id"]}/stream-input?model_id={model["model_id"]}""",
                additional_headers={
                    "xi-api-key": elevenlabs_api_key,
                },
        ) as websocket:
            websocket.send(BOS)

            # Stream text chunks and receive audio
            for text_chunk in text_chunker(text_generator):
                data = dict(text=text_chunk, try_trigger_generation=True)
                websocket.send(json.dumps(data))
                try:
                    data = json.loads(websocket.recv(1e-4))
                    if data["audio"]:
                        audio_data.append(
                            data["audio"]
                        )  # Append received audio data to list
                except TimeoutError:
                    pass

            websocket.send(EOS)

            while True:
                try:
                    data = json.loads(websocket.recv())
                    if data["audio"]:
                        audio_data.append(
                            data["audio"]
                        )  # Append received audio data to list
                except websockets.exceptions.ConnectionClosed:
                    break

            # Concatenate the audio data chunks and assign to self.chunk, use delimiter XXXXX to separate chunks
            self.chunk = audio_data

    def on_audio(self, audio: str):
        if (self.processing):
            return
        self.processing = True
        try:
            # Decoding when needed
            audio = audio.split(",")[1]
            decoded_audio = base64.b64decode(audio)

            # Read back using wavfile
            sr, audio = wavfile.read(io.BytesIO(decoded_audio))
            audio = audio.astype(np.float32)
            audio = audio / np.max(np.abs(audio))

            # Transcribe
            user_text = " ".join(seg.text
                                 for seg in transcriber.transcribe(audio)[0])
            print(user_text)
            if len(user_text) < 10:
                self.processing = False
                return

            # Add to history
            self.history.append({"role": "user", "content": user_text})

            # Generate and stream output
            model = {
                "model_id": "eleven_monolingual_v1",
            }

            text_generator = self.generate([system_prompt] +
                                           self.history[-10:])
            self.generate_stream_input(text_generator, voice, model)
            self.history.append({"role": "assistant", "content": self.answer})
            print(self.history)
            self.processing = False
        except Exception as e:
            print(e)
            self.processing = False

    def setView(self, view: str):
        self.view = view

    def setYoutubeLink(self, youtubeLink: str):
        self.youtubeLink = youtubeLink

    def setText(self, text: str):
        self.text = text

    def setImage(self, image: str):
        self.image = image
