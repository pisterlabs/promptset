#!/usr/bin/env python3

import os
import openai
import pyaudio
import io
import wave
from playsound import playsound
import boto3

# Set up OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")

# Constants
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "output.wav"
SYSTEM_MESSAGE = "I want you to act as a text based adventure game. I will give commands and you will reply with a description of what the character sees. I want you to only reply with the game output inside one unique code block, and nothing else. do not write explanations. do not type commands unless I instruct you to do so. at the start of each game, find out how many players are playing and give each of them a character. at the start of each game you should ask what type of adventure we want go on, and you should give examples. Make the game appropriate for a 9 year old. Make sure there is comedy in every game."

# Initialize PyAudio
p = pyaudio.PyAudio()

def record_audio():
    print("Recording...")

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Finished recording")

    stream.stop_stream()
    stream.close()

    # Save audio to a file
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

def translate_audio_to_text():
    audio_file = open(WAVE_OUTPUT_FILENAME, 'rb')

    response = openai.Audio.translate(model="whisper-1", file=audio_file)

    return response.text.strip()

def generate_response(user_message, messages):
    messages.append({"role": "user", "content": user_message})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    return response.choices[0].message['content'].strip()

def play_audio(text):
    polly_client = boto3.Session(
        region_name="us-east-1"  # Set the appropriate region for your use case
    ).client("polly")

    response = polly_client.synthesize_speech(
        Text=text,
        OutputFormat="mp3",
        VoiceId="Matthew"
    )

    with open("response.mp3", "wb") as f:
        f.write(response["AudioStream"].read())

    playsound("response.mp3")
    os.remove("response.mp3")

if __name__ == "__main__":
    try:
        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
        ]

        print("Assistant: Welcome to the role-playing game! I am your game master.")
        play_audio("Welcome to the role-playing game! I am your game master.")
        response = generate_response("Let's get started.", messages)
        print(f"Assistant: {response}")
        play_audio(response)

        while True:
            record_audio()
            text = translate_audio_to_text()

            print(f"You: {text}")
            response = generate_response(text, messages)

            print(f"Assistant: {response}")
            play_audio(response)
    except KeyboardInterrupt:
        print("Exiting...")
        p.terminate()
