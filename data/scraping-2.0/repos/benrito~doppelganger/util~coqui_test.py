import logging
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from dotenv import load_dotenv
import sys
import pyaudio
import wave
import time
import openai
import requests
import shutil
import subprocess
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

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

COQUI_API_TOKEN = 'lVSXFwuAHSwtYpDX8RLSDsAq0D6cMH5eKqC3dyj08ZhGW2befEvO5wuTPTsSLoQW'
voices_url = 'https://app.coqui.ai/api/v2/voices/xtts'
samples_url = 'https://app.coqui.ai/api/v2/samples/xtts'
voice_id = "6905532a-a2b1-4437-9fe6-4750571ab6ab"
text = "this is a test of ben plus tony"
timestamp = str(int(time.time()))

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
        return False

def send_to_coqui_api(text):
    global counter
    payload = {
        "speed": 1,
        "language": "en",
        "voice_id": voice_id,
        "text": text,
    }

    logging.info(f"Sending request to API with payload: {payload}")

    response = requests.post(samples_url, json=payload, headers=headers)
    logging.info(f"API response status: {response.status_code}")

    if response.status_code == 201:
        json_response = response.json()
        audio_url = json_response.get("audio_url")
        if audio_url:
            with lock:
                filename = f"working/{timestamp}_response{counter}.mp3"
                counter += 1
            if download_audio(audio_url, filename):
                logging.info(f"Downloaded audio for: {filename}")
                return filename
            else:
                logging.error(f"Error downloading audio for: {filename}")
                return None
        else:
            logging.error("No audio_url found in the API response.")
            return None
    else:
        logging.error(f"Request to API failed with status code {response.status_code}. Response content: {response.text}")
        return None

def main():
    global counter
    counter = 0

    try:
        with open(f"working/1698803415_responses.json", "r") as file:
            texts = json.load(file)
    except Exception as e:
        logging.error(f"Error reading file working/{timestamp}_responses.json. Error: {e}")
        return

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(send_to_coqui_api, text): text for text in texts}
        for future in as_completed(futures):
            text_sent = futures[future]
            try:
                future.result()
            except Exception as e:
                logging.error(f"Error processing text '{text_sent}': {e}")

if __name__ == "__main__":
    main()
