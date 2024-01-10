import io
import requests
import sounddevice as sd
import wavio
import openai
import os
from dotenv import load_dotenv

#load_dotenv('.env.example')

openai.api_key = "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"

app = Flask(__name__)
# Record audio
@app.route("/record")
def record_audio(filename, duration, fs):
    print("Recording audio...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=2)
    sd.wait()
    wavio.write(filename, recording, fs, sampwidth=2)
    print("Audio recorded and saved as", filename)

# Transcribe audio using Whisper ASR API
def transcribe_audio(filename):
    print("Transcribing audio...")
    with open(filename, "rb") as audio_file:
        transcript = openai.Audio.transcribe(
            file = audio_file,
            model = "whisper-1",
            response_format="text",
            language="en"
        )
    print(transcript)


record_audio("test.mp3", 5, 16000)
transcribe_audio("test.mp3")
