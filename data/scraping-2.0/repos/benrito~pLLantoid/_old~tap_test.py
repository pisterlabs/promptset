from dotenv import load_dotenv
import pyaudio
import wave
import time
import openai
import requests
import json
import os
import re
from collections import defaultdict
from playsound import playsound
from pydub import AudioSegment
import tempfile
import os
import sounddevice as sd
import numpy as np
import random
import nltk
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import speech_recognition as sr
import subprocess
import threading
import sounddevice as sd
import numpy as np
from scipy.io import wavfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

########################################################################################
########  Load  ########
########################################################################################

class Spinner:
    def __init__(self):
        self._running = False
        self._thread = None

    def _spin(self):
        chars = ['-', '\\', '|', '/']
        while self._running:
            for char in chars:
                sys.stdout.write(char)
                sys.stdout.flush()
                time.sleep(0.1)
                sys.stdout.write('\b')

    def start(self):
        if not self._thread or not self._thread.is_alive():
            self._running = True
            self._thread = threading.Thread(target=self._spin)
            self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join()


# Load environment variables from .env file
load_dotenv()

# Access environment variables
openai.api_key = os.environ.get("OPENAI")
eleven_labs_api_key = os.environ.get("ELEVEN")

# Load the sounds
beep_start = 'media/beep_start.mp3'
beep_stop = 'media/beep_stop.wav'
tellme = 'media/tellme.mp3'
askme = 'media/askme.mp3'
cleanse = 'media/cleanse.mp3'
ambient_sounds = [
    'media/ambient1.mp3',
    'media/ambient2.mp3',
    'media/ambient3.mp3',
    'media/ambient4.mp3'
]
acknowledgement_sounds = [
    'media/acknowledgement1.mp3',
    'media/acknowledgement2.mp3',
    'media/acknowledgement3.mp3',
    'media/acknowledgement4.mp3'
]
radiation = random.choice(ambient_sounds)
acknowledgement = random.choice(acknowledgement_sounds)


########################################################################################
########  Initiate session  ########
########################################################################################

# set the correct audio interface
def get_device_with_max_channels():
    devices = sd.query_devices()
    max_channels = 0
    max_device_id = None

    for device in devices:
        if device['max_output_channels'] > max_channels:
            max_channels = device['max_output_channels']
            max_device_id = device['index']

    if max_device_id is None:
        raise ValueError("No suitable device found.")
    
    return max_device_id

# create the seed
timestamp = str(int(time.time()))

#print the result
print("Initiating session " + timestamp)

########################################################################################
########  Revelation - recording  ########
########################################################################################

def detect_taps(threshold=10, tap_interval=0.5, taps_required=2):
    """
    Continuously monitor the microphone for taps.
    """
    tap_count = 0
    last_tap_time = time.time()

    def callback(indata, frames, pa_time, status):
        nonlocal tap_count, last_tap_time
        volume_norm = np.linalg.norm(indata) * 10
        if volume_norm > threshold:
            print("Tap detected!")  # Log message for tap detection
            if time.time() - last_tap_time < tap_interval:
                tap_count += 1
            else:
                tap_count = 1
            last_tap_time = time.time()

    with sd.InputStream(callback=callback):
        while tap_count < taps_required:
            sd.sleep(int(tap_interval * 1000))  # sleep for tap_interval seconds

    return True

if __name__ == "__main__":
    # Wait for taps
    if detect_taps():
        playsound("media/beep_start.wav")  # play a sound when taps are detected
    r = sr.Recognizer()
    time.sleep(1)  # wait for 1 second
    with sr.Microphone() as source:
        print("\n\nI'm listening...")
        audio = r.listen(source)
        with open(f"working/requests/{timestamp}_request.wav", "wb") as f:
            f.write(audio.get_wav_data())

# Use the recognizer to convert speech to text, playing some atmospherics while we wait

try:
    def ambient_background(radiation, stop_event):
        while not stop_event.is_set():
            playsound(radiation)

    # Create and start the thread
    stop_event = threading.Event()
    sound_thread = threading.Thread(target=ambient_background, args=(radiation, stop_event))
    sound_thread.daemon = True
    sound_thread.start()

    # Recognize the speech input using Google Speech Recognition
    playsound(acknowledgement)
    text = r.recognize_google(audio)
    print("\n\nI heard: " + text)

    # Prepare to pass the transcript to the prompt
    revelation = text
    filename = f"working/revelations/{timestamp}_revelation.txt"

    # Save the generated transcript to a file with epoch timestamp
    with open(f"working/revelations/{timestamp}_revelation.txt", "w") as f:
        f.write(revelation)

except sr.UnknownValueError:
    print("Google Speech Recognition could not understand audio")

except sr.RequestError as e:
    print("Could not request results from Google Speech Recognition service; {0}".format(e))

   