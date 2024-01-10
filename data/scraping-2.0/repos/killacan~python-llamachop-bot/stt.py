import pyaudio
from scipy.io.wavfile import write
import numpy as np
import keyboard
import time
import openai
import wave
import asyncio
import threading
import queue
import os
import json

with open('config.json', 'r') as f:
    config = json.load(f)

push_to_talk = config['push_to_talk']

def listen(q):
    # this function needs to record audio when a keybind is pressed
    # and then save it to a file. That file will then be sent to whisper for
    # speech to text
    print("Listening...")

    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    WAVE_OUTPUT_FILENAME = "speech.wav"
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    frames = []

    print("Recording...")

    while keyboard.is_pressed(push_to_talk):
            # record audio
            data = stream.read(CHUNK)
            frames.append(data)
   
    print("Done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    result = send_to_whisper()
    q.put(result)

async def start_listen_thread():
    q = queue.Queue()
    t = threading.Thread(target=listen, args=(q,))
    t.daemon = True
    t.start()
    result = await asyncio.get_event_loop().run_in_executor(None, q.get)
    return result

def send_to_whisper():
    # this function will send the audio file to whisper for speech to text
    # and then return the text to the main function
    print("Sending to whisper...")
    audio_file = open("speech.wav", "rb")
    try:
        response = openai.Audio.transcribe("whisper-1",audio_file)
        response['voice'] = True
        print(response)
        return response
    except:
        print("Error: Whisper is not responding. Please try again later.")
        return

