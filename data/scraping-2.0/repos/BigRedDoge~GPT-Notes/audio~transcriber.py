import openai
import wave
import pyaudio
import dotenv
import os


class Transcriber:
    """
    Transcribes audio using OpenAI's Whisper API
    args: path to recording
    """
    def __init__(self, path):
        self.path = path

    def transcribe(self, frames):
        self.save_audio(frames)
        transcript = openai.Audio.transcribe("whisper-1", open(self.path, "rb"))
        return transcript["text"]
    
    def save_audio(self, frames):
        with wave.open(self.path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
            wf.setframerate(44100)
            wf.writeframes(b''.join(frames))