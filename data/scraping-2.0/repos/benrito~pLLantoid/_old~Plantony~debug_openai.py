from dotenv import load_dotenv
import pyaudio
import wave
import time
import openai
import requests
import json
from playsound import playsound
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

# Load environment variables from .env file
load_dotenv()

# Access environment variables
openai.api_key = os.environ.get("OPENAI")
eleven_labs_api_key = os.environ.get("ELEVEN")

# The GPT-3.5 model ID you want to use
model_id = "text-davinci-003"

# The maximum number of tokens to generate in the response
max_tokens = 1024

# Construct the prompt with the embedded transcript
prompt = f"Is this working?"

# Generate the response from the GPT-3.5 model
response = openai.Completion.create(
    engine=model_id,
    prompt=prompt,
    max_tokens=max_tokens
)

# Save the response to a local file with an epoch timestamp
filename = f"responses/debug.txt"
with open(filename, "w") as f:
    f.write(response.choices[0].text)
    print(f"Output saved to responses/debug.txt")
sermon_text = response.choices[0].text
