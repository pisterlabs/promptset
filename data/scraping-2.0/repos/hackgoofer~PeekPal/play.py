from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

import pygame

# Initialize pygame mixer
pygame.mixer.init()

speech_file_path = "speech.mp3"
response = client.audio.speech.create(
    model="tts-1",
    voice="nova",
    input="Today is a wonderful day to build something people love!",
)

response.stream_to_file(speech_file_path)

# Load the MP3 music file
pygame.mixer.music.load(speech_file_path)

# Play the music
pygame.mixer.music.play()

# Wait for the music to finish playing
while pygame.mixer.music.get_busy():
    pygame.time.Clock().tick(10)
