#%%
import plotext
import numpy as np
import pyaudio
import struct
import wave
import time
from audio_get_channels import get_cur_mic
from scipy.fftpack import fft
import openai
import credentials
import os
import pyttsx3
import threading
import sys
from audio_get_channels import get_speaker
from geopy.geocoders import Nominatim
import json
from urllib.request import urlopen

script_dir = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(script_dir, "audio_output.wav")
preprompt = "You are a Ai audio guide. in the following prompt, look for the name of a city or a location, and give a one line discription of this place. Start with the location as a header."

def get_geo_location(city_name):
    # calling the Nominatim tool
    loc = Nominatim(user_agent="GetLoc")

    # entering the location name
    getLoc = loc.geocode(city_name)

    # printing address
    print(getLoc.address)

    # printing latitude and longitude
    print("Latitude = ", getLoc.latitude, "\n")
    print("Longitude = ", getLoc.longitude)

    return getLoc


def get_location():
    url = "http://ipinfo.io/json"
    response = urlopen(url)
    data = json.load(response)
    # Get location
    lng = data['loc'].split(',')[1]
    lat = data['loc'].split(',')[0]
    location = {'lat': lat, 'lng': lng}
    return location



def run_chatGPT(prompt):
    '''Run chatGPT with the prompt and return the response'''
    completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": preprompt + prompt},
        ]
    )
    answer = completion.choices[0].message.content

    return answer


def speak_answer(answer):
    engine = pyttsx3.init()
    engine.setProperty('rate', 110)
    engine.say(answer)
    engine.runAndWait()


def print_answer(answer):
    for word in answer.split():
        print(word, end=' ', flush=True)


def print_transcript(transcript):
    for word in transcript.split():
        time.sleep(0.27)
        print(word, end=' ', flush=True)


def get_transcript_whisper():
    openai.api_key = credentials.api_key
    file = open(filename, "rb")
    transcription = openai.Audio.transcribe("whisper-1", file, response_format="json")
    transcribed_text = transcription["text"]
    return transcribed_text


def audio_spectrum(num_seconds):
    script_dir = os.path.dirname(os.path.abspath(__file__))

    chunk = 2205
    channels = 1
    fs = 44100
    seconds = max(num_seconds, 0.1)
    sample_format = pyaudio.paInt16
    filename = os.path.join(script_dir, "audio_output.wav")

    print(f'\n... Recording {seconds} seconds of audio initialized ...\n')

    p = pyaudio.PyAudio()
    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    input_device_index=get_cur_mic(),
                    frames_per_buffer=chunk,
                    input=True)


    frames = []
    for i in range(0, int(fs / chunk * seconds)):
        data = stream.read(chunk, False)
        frames.append(data)


    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save the recorded data as a WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()


def run_all_functions():
    try:
        audio_spectrum(6)
    except KeyboardInterrupt:
        pass

    transcript = get_transcript_whisper()

    # If text contains one word of a stopwordlist then the script will stop
    if any(word in transcript for word in ['stop', 'Stop', 'exit', 'quit', 'end']):
        print('... Script stopped by user')
        exit()

    transcript = f' {transcript}'

    #print_transcript(transcript)

    answer = run_chatGPT(transcript)
    # Split answer into answer and location

    #print_answer(answer)

    #speak_answer(answer)

    return transcript, answer



#run_all_functions()
# ----------------------------------------------------------------
if __name__ == "__main__":
    script_start = time.time()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(script_dir, "audio_output.wav")
    run_all_functions()



# Create threads for speaking and printing the transcript
#speak_thread = threading.Thread(target=speak_answer)
#print_thread = threading.Thread(target=print_transcript)

# Start both threads
#print_thread.start()
#speak_thread.start()

# Wait for both threads to finish
#threading.wait_for(lambda: not speak_thread.is_alive()and not print_thread.is_alive())

# Wait for both threads to finish
#speak_thread.join()
#print_thread.join()
#print_answer()

# ----------------------------------------------------------------
# Restart the script


