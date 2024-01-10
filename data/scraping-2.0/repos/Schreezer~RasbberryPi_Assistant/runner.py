import pyaudio
import wave
import openai
import requests
import pygame

# Configure the OpenAI and Eleven Labs APIs
OPENAI_API_KEY = "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"
ELEVEN_API_KEY = "038461a70ead2591a29d02008134f6b8"
openai.api_key = OPENAI_API_KEY
eleven_headers = {"Authorization": f"Bearer {ELEVEN_API_KEY}"}

# Record audio and convert to text
import uuid
import pyaudio
import wave

def record_audio(seconds=5):
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 1024
    RECORD_SECONDS = seconds

    audio = pyaudio.PyAudio()

    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    frames = []

    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    audio.terminate()

    filename = f"audio_{uuid.uuid4().hex}.wav"
    with wave.open(filename, 'wb') as wave_file:
        wave_file.setnchannels(CHANNELS)
        wave_file.setsampwidth(audio.get_sample_size(FORMAT))
        wave_file.setframerate(RATE)
        wave_file.writeframes(b''.join(frames))
    
    return filename

def voice_to_text(audio_file):
    audio_file= open(audio_file, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    text = transcript['text']
    return text

def process_text_with_gpt35(text):
    model_engine = "text-davinci-003"
    response = openai.Completion.create(
        engine=model_engine,
        prompt=text,
        max_tokens=1024,
        n=1,
        temperature=0.5,
    )

    result = response.choices[0].text.strip()
    return result

import requests

import uuid

def text_to_voice(text):
    voice_id = "21m00Tcm4TlvDq8ikWAM"
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "Accept": "application/json",
        "xi-api-key": ELEVEN_API_KEY
    }
    data = {"text": text}

    response = requests.post(url, headers=headers, json=data)

    # Generate a unique file name using uuid
    output_file = f"{uuid.uuid4()}.mp3"

    with open(output_file, 'wb') as file:
        file.write(response.content)

    return output_file

def get_voices():
    url = "https://api.elevenlabs.io/v1/voices"
    headers = {
        "Accept": "application/json",
        "xi-api-key": "038461a70ead2591a29d02008134f6b8"
    }
    response = requests.get(url, headers=headers)
    return response.json()

def play_audio(audio_file):
    pygame.mixer.init()
    pygame.mixer.music.load(audio_file)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        continue

def main():
    print("Listening for voice input...")
    val = record_audio()
    play_audio(val)
    print("Converting voice to text...")
    text = voice_to_text(val)
    print(f"Input text: {text}")
    print("Processing text with GPT-3.5...")
    output = process_text_with_gpt35(text)
    print(f"Output text: {output}")
    print("Converting text to voice...")
    final = text_to_voice(output)
    print("Playing generated voice...")
    play_audio(final)

import time
import RPi.GPIO as GPIO

BUTTON_GPIO = 16

GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_GPIO, GPIO.IN, pull_up_down=GPIO.PUD_UP)
pressed = False

while True:
    # button is pressed when pin is LOW
    if not GPIO.input(BUTTON_GPIO):
        if not pressed:
            print("Button pressed!")
            pressed = True
            main()
    # button not pressed (or released)
    else:
        pressed = False
    time.sleep(0.1)
