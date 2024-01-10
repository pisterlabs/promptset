"""
This module provides a terminal assistant using OpenAI's GPT-4 model.
"""

import time
import os
import subprocess
from configparser import ConfigParser
import sys
import speech_recognition as sr
import requests
from pydub import AudioSegment
from pydub.playback import play

import openai
from gtts import gTTS

# Redirect stderr to /dev/null
sys.stderr = subprocess.DEVNULL

config = ConfigParser()
CONFIG_NAME = 'ta_auth.ini'


def create_config():
    """
    Function to create a configuration file.
    """
    # Set OpenAI API key to an empty string
    openai_key = ""
    googleapi_api_key = input("GoogleAPI key: ")
    googleapi_search_engine_id = input("GoogleAPI search engine ID: ")

    config['AUTH'] = {
        'openai': openai_key,
        'googleapi_key': googleapi_api_key,
        'googleapi_search_id': googleapi_search_engine_id
    }

    with open(CONFIG_NAME, 'w', encoding='utf-8') as config_file:
        config.write(config_file)


def check_for_config():
    """
    Function to check for a configuration file.
    """
    if os.path.exists(CONFIG_NAME):
        config.read(CONFIG_NAME)
        return

    create_config()


check_for_config()

openai.api_base = "http://localhost:1234/v1"
openai.api_key = "YOUR_API_KEY_HERE"
API_KEY = config['AUTH']['googleapi_key']
SEARCH_ENGINE_ID = config['AUTH']['googleapi_search_id']
ENDPOINT = "https://www.googleapis.com/customsearch/v1"


def ask_gpt(prompt, model="local-model"):
    """
    Function to interact with the GPT model.
    """
    response = openai.ChatCompletion.create(
        model="local-model",
        messages=[
            {"role": "system", "content": "I am your helpful assistant"},
            {"role": "user", "content": prompt}
        ],

        temperature=0.7,
    )
    return response.choices[0].message['content']


def generate_speech(text):
    """
    Function to generate speech from text using gTTS.
    """
    gtts = gTTS(text=text, lang="en-au")
    gtts.save("output.mp3")


def recognize_speech():
    """
    Function to recognize speech using Google Speech Recognition.
    """
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Speak:")
        audio = recognizer.listen(source,timeout=15, phrase_time_limit=20)

    try:
        print("Recognizing...")
        text = recognizer.recognize_google(audio)
        print("You:", text)
        return text
    except sr.UnknownValueError:
        print("Could not understand audio.")
        return ""
    except sr.RequestError as error:
        print(f"Error: {error}")
        return ""


def perform_google_search(query):
    """
    Function to perform a Google search using the Custom Search JSON API.
    """
    params = {
        'key': API_KEY,
        'cx': SEARCH_ENGINE_ID,
        'q': query
    }

    response = requests.get(ENDPOINT, params=params, timeout=5)
    search_results = response.json()

    if 'items' in search_results:
        results = search_results['items']
        for result in results:
            print(result['title'])
            print(result['link'])
            print(result['snippet'])
            print()
    else:
        print("No results found.")


def play_audio():
    """
    Function to play the speech audio.
    """
    sound = AudioSegment.from_mp3("output.mp3")
    play(sound)


def chatbot():
    """
    Main chatbot loop.
    """
    username = input("Enter your username: ")
    print(f"Hi {username}! (Type '!search' to query Google Search, \
        Press 'Enter' to respond with text input, \
        Press 'Shift+Enter' to respond with voice input, Type 'quit' to exit)")
    role = "I am a your helpful assistant. \
        I try hard to give new and interesting replies. \
        I'm also funny, witty, charming, and a great programmer. "

    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break
        if user_input.startswith('!search'):
            query = user_input[8:]
            perform_google_search(query)
            continue
        if user_input.strip() == "":
            user_input = recognize_speech()

        prompt = f"User: {user_input}\n{role}\n"
        response = ask_gpt(prompt)

        # Generate speech from the chatbot's response
        generate_speech(response)

        # Play the speech audio
        play_audio()

        print(f"{username}: {response}")

        time.sleep(3)


# Execute the chatbot
if __name__ == "__main__":
    chatbot()
