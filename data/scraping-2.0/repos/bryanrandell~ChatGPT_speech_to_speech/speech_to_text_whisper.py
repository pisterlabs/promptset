# Description: Record audio from the macOS microphone and transcribe it using OpenAI's Whisper ASR API
# really archaic with the keyboard input, but it kinda works
# tested only on macOS 10.15.7 with Python 3.9
# Bryan Randell 2023

import pyaudio
import wave
import tempfile
import openai
import os
import keyboard
import time
from dotenv import load_dotenv


# Function to record audio from the macOS microphone
def record_audio(filename):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("Press 'a' to start and stop recording.")

    frames = []
    start_time = time.time()
    while True:
        if keyboard.is_pressed('a'):
            time.sleep(0.2)
            break
        if time.time() - start_time > 100:
            print("Recording timed out.")
            return

    print("Recording...")
    start_time = time.time()

    while not keyboard.is_pressed('a'):
        if time.time() - start_time > 120:
            print("Recording is too long.")
            break
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
        time.sleep(0.01)

    print("Recording finished.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()


# Function to transcribe audio using OpenAI's Whisper ASR API
def transcribe_audio(filename: str) -> str:
    with open(filename, 'rb') as audio_file:
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
    return transcript["text"]


# Main function to record audio and transcribe it
def main_audio_to_text() -> str:
    # Record audio and save it as a temporary WAV file
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as temp_file:
        record_audio(temp_file.name)  # Record for 5 seconds

        # Use OpenAI's Whisper API to transcribe the audio
        transcription = transcribe_audio(temp_file.name)

        if transcription:
            print("Transcription:", transcription)
            return transcription
        else:
            print("Error transcribing audio.")
            return ""


if __name__ == '__main__':
    main_audio_to_text()
