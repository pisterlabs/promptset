import json
import os
import wave

import openai
import pyaudio

# read OpenAI API credentials
openai.api_key = open('apiKey.txt', 'r').read().strip()


# recording audio and saving it to file.wav
def recordAudio():
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 1024
    RECORD_SECONDS = 7
    WAVE_OUTPUT_FILENAME = "file.wav"

    audio = pyaudio.PyAudio()

    # start Recording
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    print("recording...")
    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("finished recording")

    # stop Recording
    stream.stop_stream()
    stream.close()
    audio.terminate()

    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()


# accessing file and transcribing it to english through openai api
def translateAudio():
    audioFile = open(os.path.join('file.wav'), "rb")

    transcript = openai.Audio.translate("whisper-1", audioFile)
    # Parse the JSON response
    data = json.loads(str(transcript))
    # Extract the string after the "text" key
    text_value = data["text"]
    print(text_value)
    return text_value


# complete action including recording audio and returning the transcribed audio / text for image generation
def routine():
    recordAudio()
    return translateAudio()
