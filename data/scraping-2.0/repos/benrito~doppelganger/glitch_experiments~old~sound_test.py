import logging
from dotenv import load_dotenv
import sys
import pyaudio
import wave
import time
import openai
import requests
import shutil
import os
from typing import Iterator
import json
import re
from audio_effects import OtherworldlyAudio
from collections import defaultdict
from playsound import playsound
from pydub import AudioSegment, effects
from pydub.playback import play
from pydub.generators import Sine
import tempfile
import sounddevice as sd
import numpy as np
import random
import nltk
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import speech_recognition as sr
import subprocess
import threading
from scipy.io import wavfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

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

def reencode_mp3(filename):
    temp_filename = "temp_reencoded.mp3"
    cmd = [
        'ffmpeg',
        '-i', filename,
        '-codec:a', 'libmp3lame',
        '-qscale:a', '2',
        temp_filename
    ]
    subprocess.run(cmd, check=True)
    # Overwrite the original file with the re-encoded version
    subprocess.run(['mv', temp_filename, filename], check=True)

def route_sound(filename, device_id, channel):
    # First, re-encode the mp3 file to ensure it has a correct header
    reencode_mp3(filename)

    # Load audio file using pydub
    sound = AudioSegment.from_file(filename, format="mp3")

    # Convert sound to numpy array and normalize
    samples = np.array(sound.get_array_of_samples()).astype(np.float32) / (2**15)
    
    max_output_channels = sd.query_devices(device_id)['max_output_channels']

    if channel >= max_output_channels:
        raise ValueError(f"The device only has {max_output_channels} output channel(s).")

    # Create an empty array with the correct number of output channels
    zeros = np.zeros((len(samples), max_output_channels), dtype=np.float32)

    # Copy the mono audio data to the desired channel
    zeros[:, channel] = samples

    # Start a stream and play it
    print(f"\n\nRouting {filename} to device {device_id} on channel {channel}")
    with sd.OutputStream(device=device_id, channels=max_output_channels, samplerate=sound.frame_rate) as stream:
        stream.write(zeros)

def list_files_with_same_timestamp(directory):
    # List all files in the directory
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    
    # Use regex to extract timestamp and counter from filenames
    pattern = rf"(\d+)_response(\d+).mp3"

    # Group files by timestamp
    timestamp_files = defaultdict(list)
    for file in files:
        match = re.match(pattern, file)
        if match:
            timestamp, counter = match.groups()
            timestamp_files[timestamp].append((int(counter), file))

    return timestamp_files

def main():
    directory = f"working/"
    device_id = get_device_with_max_channels()  # Automatically get the device ID with the most channels
    target_timestamp = "1698810220"
    timestamp_files = list_files_with_same_timestamp(directory)
    if target_timestamp not in timestamp_files:
        print(f"No files with timestamp {target_timestamp} found.")
        return

    files = timestamp_files[target_timestamp]
    
    # Sort the files based on counter for a deterministic order
    sorted_files = sorted(files, key=lambda x: x[0])
    
    max_output_channels = sd.query_devices(device_id)['max_output_channels']

    threads = []  # To store thread objects

    for index, (counter, filename) in enumerate(sorted_files):
        # Cycle through the number of available channels
        channel = index % max_output_channels

        # Create a thread for each sound and start it immediately
        t = threading.Thread(target=route_sound, args=(os.path.join(directory, filename), device_id, channel))
        threads.append(t)
        t.start()

        # Sleep for 1.25 seconds before starting the next thread
        time.sleep(1.25)

    # Wait for all threads to complete
    for t in threads:
        t.join()

    print("Tap again...")

if __name__ == "__main__":
    main()