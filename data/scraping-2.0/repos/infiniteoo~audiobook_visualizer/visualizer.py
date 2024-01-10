import os
from dotenv import load_dotenv
import time
from openai import OpenAI
import speech_recognition as sr
from PIL import Image
import matplotlib.pyplot as plt
from pydub import AudioSegment
import requests
from io import BytesIO
import threading
import pygame

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)  # Replace with your OpenAI API key

chunk_duration = 10  # seconds

def transcribe_audio(audio_file):
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 4000

    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        print("Speech recognition could not understand audio")
        return ""
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return ""

def get_current_audio_chunk(audio_file, start_time=0, end_time=10):
    audio = AudioSegment.from_file(audio_file)
    chunk = audio[start_time * 1000:end_time * 1000]
    chunk.export("current_chunk.wav", format="wav")
    return "current_chunk.wav"

def generate_and_save_image(prompt, save_path):
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",  # Adjust the size based on what is supported by the model
        quality="standard",
        n=1,
    )

    image_url = response.data[0].url
    image_data = requests.get(image_url).content

    with open(save_path, 'wb') as image_file:
        image_file.write(image_data)

    img = Image.open(BytesIO(image_data))
    plt.imshow(img)
    plt.axis('off')
    plt.pause(0.001)  # Add a small pause to allow the script to continue


def display_images_concurrently(audio_file, start_time, end_time):
    audio_thread = threading.Thread(target=play_audio, args=(audio_file,))
    audio_thread.start()

    while end_time <= len(AudioSegment.from_file(audio_file)) / 1000:
        current_audio_chunk = get_current_audio_chunk(audio_file, start_time, end_time)
        transcribed_text = transcribe_audio(current_audio_chunk)

        if transcribed_text:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            save_path = f"./images/generated_image_{timestamp}.png"
            generate_and_save_image(transcribed_text, save_path)

        start_time = end_time
        end_time += chunk_duration

        time.sleep(chunk_duration)

def play_audio(audio_file):
    pygame.mixer.init()
    pygame.mixer.music.load(audio_file)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

if __name__ == "__main__":
    audio_file = "book.mp3"
    start_time = 0
    end_time = 10
    display_images_concurrently(audio_file, start_time, end_time)
