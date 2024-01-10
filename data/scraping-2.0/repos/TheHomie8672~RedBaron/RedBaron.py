import concurrent.futures
import os
import pyttsx3
import random
import requests
import speech_recognition as sr
import time
import openai
import webbrowser
import validators
import pyaudio
import startup_sequence1
import WeatherAPI
import wave
from openai import *
from startup_sequence1 import *
from WeatherAPI import *
from RedBaronGraphics import *
from playsound import playsound
from gtts import gTTS
from WeatherAPI import get_weather_data
from WebSearchFUNC import search
from WebSearchFUNC import say
import speech_recognition as sr
from io import BytesIO
from PIL import Image

openai.api_key = "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"

prompt1 = "You are RedBaron, a coding and general inquiry virtual assistant. You should avoid long winded responses. You were created by the Programmer Marcus Sherman."


 
# Initialize text-to-speech engine
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

# Initialize speech recognizer
r = sr.Recognizer()

def get_input(prompt, is_keyword=False, mute=False):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        if not mute:
            print(prompt)
        if is_keyword:
            r.energy_threshold = 4500
            r.adjust_for_ambient_noise(source, duration=0.5)
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return ""



def play_audio(audio_file):
    CHUNK = 1024
    wf = wave.open(audio_file, 'rb')
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)
    data = wf.readframes(CHUNK)
    while data:
        stream.write(data)
        data = wf.readframes(CHUNK)
    stream.stop_stream()
    stream.close()
    p.terminate()

def on_phrase(self, recognizer, audio):
    try:
        recognized_text = recognizer.recognize_google(audio)
        print("You said:", recognized_text)
        if "search" in recognized_text:
            query = recognized_text.split("search", 1)[1].strip()
            search(query)
    except sr.UnknownValueError:
        pass
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))


def search(query):
    url = f"https://www.google.com/search?q={query}"
    webbrowser.open(url)

def callback(recognizer_instance, audio, mute_flag=False):
    try:
        if not mute_flag:
            recognizer_instance.stop()
    except sr.UnknownValueError:
        pass
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))

import webbrowser

def say_output(output_text):
    engine = pyttsx3.init()
    engine.say(output_text)
    engine.runAndWait()
    





start_up_seq_main_loop()
def main_loop():
    waiting_for_keyword = True
    flag_state = False
    flag_time = None
    input_text = ""
    mute_flag = True
    user_input = ""
    response_text = ""

    while True:
        if waiting_for_keyword:
            response_text = say_output(response_text)  # Assign the return value to response_text
            
            input_text = get_input("Say 'Hey red' to start a conversation.", True)

            if input_text is not None and any(keyword in input_text.lower() for keyword in ["hey red", "hey read", "he read", "hey rea"]):
                waiting_for_keyword = False
                say_output(response_text)
        else:
            # Get the user's input
            user_input = get_input("", False, mute_flag)

            # Check if the user has stopped speaking
            if not user_input:
                if flag_state and time.time() - flag_time >= 5:
                    waiting_for_keyword = True
                    flag_state = False
                    say_output("Say , Hey red , If you need me! ") 
                continue

            # Process the user's input
            print(f"User input: {user_input}")
            if "exit" in user_input.lower() or "goodbye" in user_input.lower() or "bye" in user_input.lower():
                say_output("Goodbye!")
                return

            elif "weather" in user_input.lower():
                city = user_input.split()[-1]
                print(f"Getting weather data for {city}...")
                weather_data = get_weather_data(city, os.environ.get("OPENWEATHER_API_KEY"))

                if weather_data:
                    temperature = weather_data["temperature"]
                    weather_description = weather_data["weather_description"]
                    response_text = f"The temperature in {city} is {temperature:.1f} degrees Celsius with {weather_description}."
                    say_output(response_text)
                    print(response_text)
                else:
                    response_text = f"Sorry, could not retrieve weather data for {city}."
                    say_output(response_text)
                    print(response_text)

            elif "mute" in user_input.lower():
                mute_flag = True
                say_output("Okay, I'll stop talking now.")

            elif "unmute" in user_input.lower():
                mute_flag = False
                say_output("Okay, I'm back.")


            else:
                response_text = ""
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        openai.Completion.create,
                        engine="text-davinci-003",
                        prompt=prompt1 + user_input,
                        max_tokens=250,
                        n=1,
                        stop=None,
                        temperature=0.3
                    )
                    response_text = future.result().choices[0].text.strip()
                    

                print(response_text)

        if "search" in user_input.lower():
            query = user_input.replace("search", "").strip()
            say_output(f"Searching for {query}...")
            search(query)

        # Check if response text is a valid URL
        elif validators.url(str(response_text)):
            webbrowser.open(response_text)

        else:
            
            say_output(response_text)
            flag_state = True
            flag_time = time.time()

if __name__ == "__main__":
    main_loop()
