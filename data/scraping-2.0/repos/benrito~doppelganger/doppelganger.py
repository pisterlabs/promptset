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
import soundfile as sf
import numpy as np
import random
import speech_recognition as sr
import subprocess
import threading
from scipy.io import wavfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Load environment variables from .env file
load_dotenv()

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")

# Access environment variables
openai.api_key = os.environ.get("OPENAI")
COQUI_API_TOKEN = 'lVSXFwuAHSwtYpDX8RLSDsAq0D6cMH5eKqC3dyj08ZhGW2befEvO5wuTPTsSLoQW'
voices_url = 'https://app.coqui.ai/api/v2/voices/xtts'
samples_url = 'https://app.coqui.ai/api/v2/samples/xtts'

# Load the sounds
beep_start = 'media/beep_start.mp3'
beep_stop = 'media/beep_stop.wav'
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

timestamp = str(int(time.time()))
print(timestamp)

print("\n\nInitiating session " + timestamp)

print("\n\nTap away")

########################################################################################
########  Main loop  ########
########################################################################################

def loop():
    def detect_taps(threshold=10, tap_interval=1, taps_required=3):
        """
        Continuously monitor the microphone for taps.
        """
        tap_count = 0
        last_tap_time = time.time()

        def callback(indata, frames, pa_time, status):
            nonlocal tap_count, last_tap_time
            volume_norm = np.linalg.norm(indata) * 10
            if volume_norm > threshold:
                print("\n\nTap detected!")  # Log message for tap detection
                if time.time() - last_tap_time < tap_interval:
                    tap_count += 1
                else:
                    tap_count = 1
                last_tap_time = time.time()

        with sd.InputStream(callback=callback):
            while tap_count < taps_required:
                sd.sleep(int(tap_interval * 1000))  # sleep for tap_interval seconds

        return True

    MAX_ATTEMPTS = 1
    attempts = 0

    r = sr.Recognizer()

    while attempts < MAX_ATTEMPTS:
        # Wait for taps
        if detect_taps():
            playsound(beep_start)
        time.sleep(.25)
        with sr.Microphone() as source:
            print("\n\nI'm listening...")
            audio = r.listen(source)
            with open(f"working/{timestamp}_recording.wav", "wb") as f:
                f.write(audio.get_wav_data())

        # Use the recognizer to convert speech to text, playing some atmospherics while we wait
        try:

            def mirror_recording():
                mirror = OtherworldlyAudio(f"working/{timestamp}_recording.wav")
                time.sleep(3)  # wait for n second
                mirror.pitch_shift()
                mirror.play_sound()

            def ambient_background(radiation, stop_event):
                # while not stop_event.is_set():
                playsound(radiation)

            # Create and start the thread
            stop_event = threading.Event()
            sound_thread = threading.Thread(target=ambient_background, args=(radiation, stop_event))
            mirror_thread = threading.Thread(target=mirror_recording)
            # download_thread = threading.Thread(target=download_audio)
            sound_thread.daemon = True
            sound_thread.start()

            # Recognize the speech input using Google Speech Recognition
            playsound(acknowledgement)
            mirror_thread.start()

            # TODO: add spinner
            text = r.recognize_google(audio)
            print("\n\nI heard: " + text)

            # Prepare to pass the transcript to the prompt
            revelation = text

            with open(f"working/{timestamp}_recording.txt", "w") as f:
                f.write(revelation)

            break  # Exit the loop since we successfully got the transcript

        except sr.UnknownValueError:
            print("\n\nI couldn't understand you. Tap again.")
            playsound(beep_stop)
            stop_event.set()  # Stop the radiation sound
            attempts += 1
            if attempts == MAX_ATTEMPTS:
                playsound(beep_stop)
                sys.exit(1)

        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))
            stop_event.set()  # Stop the radiation sound
            attempts += 1
            if attempts == MAX_ATTEMPTS:
                playsound(beep_stop)
                sys.exit(1)

    ########################################################################################
    ########  Revelation — prompt  ########
    ########################################################################################

    # The GPT-3.5 model ID you want to use
    model_id = "gpt-3.5-turbo-instruct"

    # The maximum number of tokens to generate in the response
    max_tokens = 1024

    # Come up with responses
    prompt = f"I am engaging in autohypnosis. I will probe more deeply into what I may be repressing, three different ways, always in first-person: \n\n'{revelation}.'\n\n Respond with a simple JSON array, strings only:"
    # print("Prompt:\n\n"prompt)
    response = openai.Completion.create(
        engine=model_id,
        prompt=prompt,
        max_tokens=max_tokens
    )

    # Save the responses to a local file with an epoch timestamp
    filename = f"working/{timestamp}_responses.json"
    with open(filename, "w") as f:
        f.write(response.choices[0].text)
    responses = response.choices[0].text
    print("\n\nResponses:" + responses)

    ########################################################################################
    ########  Voice cloning  ########
    ########################################################################################

    # Upload sample to get the voice ID
    headers = {
        'accept': 'application/json',
        'Authorization': f'Bearer {COQUI_API_TOKEN}'
    }

    files = {
        'files': (f'working/{timestamp}_recording.wav', open(f'working/{timestamp}_recording.wav', 'rb'))
    }

    response = requests.post(voices_url, headers=headers, files=files)

    if response.status_code == 201:
        response_json = response.json()
        voice_id = response_json.get('voice', {}).get('id')
        if voice_id:
            print(f'Voice ID: {voice_id}')
        else:
            print('Voice ID not found in the response.')
    else:
        print(f'Error uploading test.wav: {response.status_code}')


########################################################################################
########  Audio generation  ########
########################################################################################

    headers = {
        "accept": "application/json",
        "authorization": f"Bearer {COQUI_API_TOKEN}",
        "content-type": "application/json",
    }

    lock = Lock()

    def download_audio(audio_url: str, filename: str) -> bool:
        try:
            response = requests.get(audio_url, stream=True)
            response.raise_for_status()
            with open(filename, 'wb') as out_file:
                shutil.copyfileobj(response.raw, out_file)
            return True
        except requests.RequestException:
            logging.error("Error while downloading audio.")
            return False

    def send_to_coqui_api(text):
        global counter
        payload = {
            "speed": 1,
            "language": "en",
            "voice_id": voice_id,
            "text": text,
        }

        response = requests.post(samples_url, json=payload, headers=headers)
        logging.info(f"Response from uploading: {response.status_code}, {response.text}")
        if response.status_code == 201:
            json_response = response.json()
            audio_url = json_response.get("audio_url")
            if audio_url:
                with lock:
                    filename = f"working/{timestamp}_response{counter}.wav"
                    counter += 1
                if download_audio(audio_url, filename):
                    return filename
                else:
                    logging.error("Error during audio download.")
                    return None
            else:
                logging.error("Audio URL not found in the response.")
                return None
        else:
            logging.error(f"Unexpected response status code: {response.status_code}")
            return None

    def parallel_process():
        global counter
        counter = 0

        try:
            with open(f"working/{timestamp}_responses.json", "r") as file:
                texts = json.load(file)
        except Exception as e:
            return

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(send_to_coqui_api, text): text for text in texts}
            for future in as_completed(futures):
                text_sent = futures[future]
                try:
                    future.result()
                except Exception as e:
                    pass

    parallel_process()

    ########################################################################################
    ########  Audio playback  ########
    ########################################################################################

    # Wait for the mirror sound to have played
    mirror_thread.join()

    # def route_sound(filename, device_id, channel):

    #     # Load audio file using pydub
    #     sound = AudioSegment.from_file(filename, format="mp3")

    #     # Convert sound to numpy array and normalize
    #     samples = np.array(sound.get_array_of_samples()).astype(np.float32) / (2**15)
        
    #     max_output_channels = sd.query_devices(device_id)['max_output_channels']

    #     if channel >= max_output_channels:
    #         raise ValueError(f"The device only has {max_output_channels} output channel(s).")

    #     # Create an empty array with the correct number of output channels
    #     zeros = np.zeros((len(samples), max_output_channels), dtype=np.float32)

    #     # Copy the mono audio data to the desired channel
    #     zeros[:, channel] = samples

    #     # Start a stream and play it
    #     print(f"\n\nRouting {filename} to device {device_id} on channel {channel}")
    #     with sd.OutputStream(device=device_id, channels=max_output_channels, samplerate=sound.frame_rate) as stream:
    #         stream.write(zeros)

    # Wait for the audio generation to finish
    # download_thread.join()

    def list_files_with_same_timestamp(directory):
        # List all files in the directory
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        
        # Use regex to extract timestamp and counter from filenames
        pattern = rf"(\d+)_response(\d+).wav"

        # Group files by timestamp
        timestamp_files = defaultdict(list)
        for file in files:
            match = re.match(pattern, file)
            if match:
                timestamp, counter = match.groups()
                timestamp_files[timestamp].append((int(counter), file))

        return timestamp_files

    def showtime():
        directory = "working/"
        timestamp_files = list_files_with_same_timestamp(directory)

        if timestamp not in timestamp_files:
            print(f"No files with timestamp {timestamp} found.")
            #wait two seconds, try again, a maximum of four times
            return

        sorted_files = sorted(timestamp_files[timestamp], key=lambda x: x[0])

        for _, filename in sorted_files:
            filepath = os.path.join(directory, filename)
            data, samplerate = sf.read(filepath)
            device_info = sd.query_devices(sd.default.device['output'], 'output')
            num_channels = device_info['max_output_channels']

            if data.ndim == 1:  # Check if it's mono
                data = np.tile(data[:, np.newaxis], (1, num_channels))

            sd.play(data, 24000)
            sd.wait()
            time.sleep(.5)

        print("Tap again...")

    if __name__ == "__main__":
        showtime()
        playsound(beep_stop)
        stop_event.set()  # Stop the radiation sound

if __name__ == "__main__":
    try:
        while True:
            loop()
    except KeyboardInterrupt:  # Press Ctrl+C to exit the loop
        print("\nScript terminated by user.")
