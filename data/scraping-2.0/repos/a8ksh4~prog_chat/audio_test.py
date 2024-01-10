#!/usr/bin/env python3
import sys
import select
import time
import pyaudio
import wave
import openai

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 16000
openai.api_key = sys.environ['OPENAI_API_KEY']
filename = '/tmp/foo.wav'

def is_enter_pressed():
    # Check if there is data available to read on sys.stdin (file descriptor 0)
    rlist, _, _ = select.select([sys.stdin], [], [], 0)
    return sys.stdin in rlist

def flush_input_buffer():
    while is_enter_pressed():
        sys.stdin.read(1)


while True:
    # Your main program loop
    with open(filename, 'w') as wf:
        pass

    print('Press enter to start recording...')
    while not is_enter_pressed():
        time.sleep(0.1)
    print('Recording...  Press enter again to stop.')
    flush_input_buffer()

    # Record audio

    with wave.open(filename, 'wb') as wf:
        p = pyaudio.PyAudio()
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)

        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True)
        while not is_enter_pressed():
            wf.writeframes(stream.read(CHUNK))
        stream.stop_stream()
        stream.close()
        p.terminate()

    flush_input_buffer()
    with open(filename, "rb") as audio_file:
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
        print(transcript)


    time.sleep(1)