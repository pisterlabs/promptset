
import requests
import os
import openai
import io
from elevenlabs import generate, play, voices, set_api_key
import pyperclip
import pygetwindow as gw
import win32process
from asyncio import wait_for
import threading
import time
from turtle import st
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import os
from pathlib import Path
import datetime
from rest import process_files
from processor import Ctx
import numpy as np
import sounddevice as sd
import webrtcvad
from queue import Queue



start_time = datetime.datetime.now()
data_dir = Path(os.getenv('LOG_DIR')) / 'notes' / datetime.date.today().strftime("%Y-%m-%d")
data_dir.mkdir(parents=True, exist_ok=True)

def process_text(predicted_text, ctx):
    t = datetime.datetime.now().strftime("%HH%MM%SS")
    with (data_dir/f'rec{t}.wav').open('wb') as f:
        f.write(ctx.audio_data.getbuffer())
    with (data_dir.parent/'notes.log').open('w+t') as f:
        f.write(f'{t}\t{predicted_text}\n')
    return predicted_text

