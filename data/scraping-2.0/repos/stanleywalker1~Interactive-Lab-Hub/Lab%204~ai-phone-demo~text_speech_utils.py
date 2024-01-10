# text_speech_utils.py

import openai
import sounddevice as sd
import audiofile as af
from scipy.io.wavfile import write
from gtts import gTTS

import multiprocessing
import pyttsx3
import keyboard

import requests

from myapikeys import ELEVENLABS_KEY

# API constants
API_URL = "https://api.elevenlabs.io/v1/text-to-speech/<voice-id>/stream"
API_HEADERS = {
    "Accept": "audio/mpeg",
    "Content-Type": "application/json",
    "xi-api-key": ELEVENLABS_KEY  # Using the imported key
}

def say(text):
    audio_filename = "temp_speech_output.mp3"
    myobj = gTTS(text=text, lang='en', slow=False)
    myobj.save(audio_filename)
    play_audio(audio_filename)

def record_audio(filename, sec, sr = 44100):
    audio = sd.rec(int(sec * sr), samplerate=sr, channels=1, blocking=False)
    sd.wait()
    write(filename, sr, audio)

def record_audio_manual(filename, sr = 44100):
    input("  ** Press enter to start recording **")
    audio = sd.rec(int(10 * sr), samplerate=sr, channels=1)
    input("  ** Press enter to stop recording **")
    sd.stop()
    write(filename, sr, audio)

def play_audio(filename):
    signal, sr = af.read(filename)
    sd.play(signal, sr)

def transcribe_audio(filename):
    audio_file= open(filename, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    audio_file.close()
    return transcript

def translate_audio(filename):
    audio_file= open(filename, "rb")
    translation = openai.Audio.translate("whisper-1", audio_file)
    audio_file.close()
    return translation

def save_text_as_audio(text, audio_filename):
    voice_id = "21m00Tcm4TlvDq8ikWAM"  # Replace with your preferred voice ID.
    model_id = "eleven_monolingual_v1"  # Default model ID.
    payload = {
        "text": text,
        "model_id": model_id,
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }
    
    # Call the Text to Speech API
    response = requests.post(API_URL.replace("<voice-id>", voice_id), headers=API_HEADERS, json=payload, stream=True)
    CHUNK_SIZE = 1024
    with open(audio_filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                f.write(chunk)
