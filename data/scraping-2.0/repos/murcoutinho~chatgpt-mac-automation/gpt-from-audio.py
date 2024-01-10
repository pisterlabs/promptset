#!/usr/bin/env python3

import sys
import openai
import pyaudio
import wave
from pynput import keyboard
import threading
import time

def record_audio(output_filename, channels=1, rate=44100, chunk=1024, format=pyaudio.paInt16, max_duration=10):
    audio = pyaudio.PyAudio()

    stream = audio.open(format=format,
                        channels=channels,
                        rate=rate,
                        input=True,
                        frames_per_buffer=chunk)

    frames = []
    recording = True
    start_time = time.time()

    listener = keyboard.Listener(on_press=None)  # Initialize the listener outside the function

    def on_press(key):
        nonlocal recording, listener
        if key == keyboard.Key.space:
            recording = False
            listener.stop()  # Stop the listener when the space key is pressed

    listener.on_press = on_press  # Assign the on_press function

    space_key_thread = threading.Thread(target=listener.start)  # Start the listener thread
    space_key_thread.start()

    while recording:
        elapsed_time = time.time() - start_time
        if elapsed_time >= max_duration:
            recording = False
            listener.stop()  # Stop the listener when the maximum duration is reached
            break
        data = stream.read(chunk)
        frames.append(data)

    space_key_thread.join()  # Wait for the space_key_thread to finish

    stream.stop_stream()
    stream.close()
    audio.terminate()
    
    with wave.open(output_filename, 'wb') as wave_file:
        wave_file.setnchannels(channels)
        wave_file.setsampwidth(audio.get_sample_size(format))
        wave_file.setframerate(rate)
        wave_file.writeframes(b''.join(frames))

def main():
    # Load your API key from an environment variable or secret management service
    openai.api_key = "{KEYPLACEHOLDER}"

    record_audio("output.wav");
    
    audio_file= open("output.wav", "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)

    if len(sys.argv) > 1:
        question = transcript['text']
        response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                        {"role": "system", "content": "You are a helpfull assistant."},
                        {"role": "user", "content": question},
                    ]
            )
        print(response['choices'][0]['message']['content'])
    else:
        print(transcript['text'])

if __name__ == '__main__':
    main()

