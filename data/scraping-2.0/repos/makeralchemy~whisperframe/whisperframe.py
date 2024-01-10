

import sounddevice as sd
import soundfile as sf
import openai
from PIL import Image, ImageDraw, ImageFont
import toml
import webbrowser
import json
import requests
import pyperclip
from datetime import datetime
from io import BytesIO
import os



# record 5 minutes of audio and write to file  output.wav
def record_audio(filename, duration_mins=5):
    duration = duration_mins * 60  # 5 minutes in seconds
    sample_rate = 44100  # CD quality audio
    channels = 2  # stereo

    # Record audio
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels)

    # Wait for recording to finish
    sd.wait()

    # Save recording to file
    sf.write(filename, recording, sample_rate)

def transcribe_audio(api_key, filename):
    # Load audio file
    audio, sample_rate = sf.read(filename)

    # Convert audio to text using OpenAI API
    openai.api_key = api_key
    response = openai.Completion.create(
        engine="davinci",
        prompt=f"Transcribe the following audio:\n{audio.tolist()}",
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.7,
    )

    # Extract transcription from API response
    transcription = response.choices[0].text.strip()

    return transcription

def summarize_text(api_key, text):
    # Summarize text using OpenAI API
    openai.api_key = api_key
    response = openai.Completion.create(
        engine="davinci",
        prompt=f"Summarize the following text:\n{text}",
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.7,
    )

    # Extract summary from API response
    summary = response.choices[0].text.strip()

    return summary


def generate_image(api_key, text):
    # Generate image using OpenAI API
    openai.api_key = api_key
    response = openai.Image.create(
        # prompt=f"Create an image based on the following text:\n{text}",
        prompt=f"{text}",
        n=1,
        size="512x512",
        response_format="url",
    )

    url = response["data"][0]["url"]

    
    # response_dict = json.loads(json.dumps(response))
    # url = response_dict["data"][0]["url"]

    return url


def save_image_from_url(url, filename, folder):
    # Get current date and time
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    # Append date and time to filename
    filename_with_timestamp = f"{filename}_{timestamp}.png"

    # Create full path for the new file
    full_path = os.path.join(folder, filename_with_timestamp)

    # Download image from URL
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))

    # Save image to file
    image.save(full_path)

    return full_path


def display_image_in_browser(url):
    webbrowser.open(url)


def get_toml_key(filename, section, key):
    with open(filename, "r") as f:
        data = toml.load(f)
    return data[section][key]


def copy_to_clipboard(text):
    pyperclip.copy(text)

print("Querying audio devices...")
print(sd.query_devices())

# print("Recording audio...")
# record_audio("output.wav", duration_mins=1)

print("Getting OpenAI API key...")
filename = "api_keys.toml"
section = "api"
key = "openai"
api_key = get_toml_key(filename, section, key)
# print(filename, ':', section, ':', key, ':', api_key)

print("Generating image...")
url = generate_image(api_key, "Monkeys playing poker at midnight")
# print(url)

print("Copying URL to clipboard...")
copy_to_clipboard(url)

print("Saving image...")
print(save_image_from_url(url, "output", "images"))

print("Displaying image in browser...")
display_image_in_browser(url)
