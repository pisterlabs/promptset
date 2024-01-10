from dotenv import load_dotenv
import traceback
import sys
import io
import logging
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
import speech_recognition as sr
import numpy as np
import random
import nltk
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import subprocess
import threading
from pydub import AudioSegment, effects
from pydub.playback import play
from pydub.generators import Sine
from io import BytesIO

import TonySpeech

# TODO — time it

chatter_finished_event = threading.Event()

load_dotenv()
openai.api_key = os.environ.get("OPENAI")   
eleven_labs_api_key = os.environ.get("ELEVEN")

def wait_for_wake_word(r):
    # global all_threads
    greeting_sounds = ['global/media/greeting1.mp3', 'global/media/greeting2.mp3', 'global/media/greeting3.mp3', 'global/media/greeting4.mp3']
    greeting = random.choice(greeting_sounds)

    logging.info("Waiting for wake word...")
    wake_phrases = ["Ready", "Reading", "Red tea", "Rudy", "Ton", "Tone", "Listening", "Activate", "Rody", "Leaving", "Heavy", "Tony", "Danny", "Wake", "Ruddy"]

    with sr.Microphone() as source:

        last_time_heard = time.time()  # Initialize the timer when you start listening

        while True:
            text = ""
            
            # Listen for speech and store it as audio data
            print("I'm listening...")
            r.adjust_for_ambient_noise(source)
            audio = r.listen(source, 10, 3)
            # with open(f"working/{current_mode}/recordings/{timestamp}_recording.wav", "wb") as f:
            #     f.write(audio.get_wav_data())

            try:
                # Recognize the speech input using Google Speech Recognition
                text = r.recognize_google(audio)

            except sr.UnknownValueError:
                pass # print("Google Speech Recognition could not understand audio")

            except sr.RequestError as e:
                print("\n\nCould not request results from Google Speech Recognition service; {0}".format(e))

            if text:
                print("I heard: " + text)
                last_time_heard = time.time()  # Reset the timer when you detect speech

                for phrase in wake_phrases:
                    if phrase.lower() in text.strip().lower():
                        print(f"\n\nWake phrase detected!")
                        playsound(greeting)  # Play the greeting sound
                        activate_chatter(r)
                        # chatter_finished_event.set()  # TODO: Signal that activate_chatter has finished
                        return
            
            # Check the time since the last time speech was detected
            current_time = time.time()
            if current_time - last_time_heard >= 600:
                print("Noone's here. Quitting...")
                sys.exit()

def get_device_with_max_channels():
    devices = sd.query_devices()
    max_channels = 0
    max_device_id = None

    for device in devices:
        if device['max_output_channels'] > max_channels:
            max_channels = device['max_output_channels']
            max_device_id = device['index']
            # print(f"We have {max_channels} channels")

    if max_device_id is None:
        raise ValueError("No suitable device found.")
    
    return max_device_id, max_channels
    
# load the current mode
def get_mode_data():
    # Load the JSON data from the file
    with open("modes/modes.json", "r") as file:
        parsed_data = json.load(file)

    # Select a random mode from the parsed JSON data
    random_mode_key = random.choice(list(parsed_data["MODES"].keys()))
    random_mode = parsed_data["MODES"][random_mode_key]

    # Save the current mode to a new JSON file
    with open("working/current_mode.json", "w") as file:
        json.dump(random_mode, file, indent=4)

    # Print selected mode for testing purposes
    print(f"\n\nCurrent mode: {random_mode['current_mode']}")
    print(f"\n\nQuestion: {random_mode['description']}")
    playsound(f"modes/{random_mode['current_mode']}/beep_start.mp3")
    time.sleep(1)

    try:
        with open('working/current_mode.json', 'r') as file:
            shared_dict = json.load(file)
            
            current_mode = shared_dict.get('current_mode', None)
            responding_prompt1 = shared_dict.get('responding_prompt1', None)
            if current_mode:
                directory_path = os.path.join('working', current_mode)
                subdirs = ['recordings', 'media', 'audio', 'analysis']

                for subdir in subdirs:
                    subdir_path = os.path.join(directory_path, subdir)
                    if not os.path.exists(subdir_path):
                        os.makedirs(subdir_path)

            return current_mode, responding_prompt1
            
    except FileNotFoundError:
        return None, None

# Load character configurations TODO: clean this up
def load_character_configs(filename="working/current_characters.json"):
    with open(filename, "r") as file:
        config = json.load(file)
        return config["characters"]

def load_characters_from_config(api, config_file="working/current_characters.json"):
    character_configs = load_character_configs(config_file)
    
    for character_data in character_configs:
        character = Character(
            character_data["name"],
            character_data["description"],
            character_data["system_message"],
            character_data["default_channel"],
            character_data["eleven_voice_id"]
        )
        api.assign_character(character)

def run_activate_chatter(r):
    t = threading.Thread(target=activate_chatter, args=(r,))
    t.start()
    t.join()  # This will block until the thread t finishes

def activate_chatter(r):
    time.sleep(1)
    current_mode, responding_prompt1 = get_mode_data()
    timestamp = str(int(time.time()))
    print("\n\nChatter activated!")
    api = ChannelAPI(current_mode=f"{current_mode}")

    problem_sounds = ['global/media/problem1.mp3', 'global/media/problem2.mp3', 'global/media/problem3.mp3', 'global/media/problem4.mp3', 'global/media/problem5.mp3']
    fail_sounds = ['global/media/fail1.mp3', 'global/media/fail2.mp3', 'global/media/fail3.mp3', 'global/media/fail4.mp3', 'global/media/fail5.mp3', 'global/media/fail6.mp3', 'global/media/fail7.mp3', 'global/media/fail8.mp3', 'global/media/fail9.mp3']
    ack_sounds = ['global/media/acknowledgement1.mp3', 'global/media/acknowledgement2.mp3', 'global/media/acknowledgement3.mp3', 'global/media/acknowledgement4.mp3']
    ambient_sounds = ['global/media/ambient1.mp3', 'global/media/ambient2.mp3', 'global/media/ambient3.mp3']
    greeting_sounds = ['global/media/greeting1.mp3', 'global/media/greeting2.mp3', 'global/media/greeting3.mp3', 'global/media/greeting4.mp3']
    problem = random.choice(problem_sounds)
    fail = random.choice(fail_sounds)
    ack = random.choice(ack_sounds)
    ambient = random.choice(ambient_sounds)
    greeting = random.choice(greeting_sounds)

    # with sr.Microphone() as source:
    MAX_RETRIES = 3
    retry_count = 0
    while retry_count < MAX_RETRIES:
        	# SEND TO ARDUINO: listening
		print("\n\nI'm listening...")
            
    #        try:
    #            audio = r.listen(source, 100, 10)  # Capture new audio
    #            text = r.recognize_google(audio)  # Try to recognize it
    #            with open(f"working/{current_mode}/recordings/{timestamp}_recording.wav", "wb") as f:
    #                f.write(audio.get_wav_data())


		TonySpeech.recoSppech(TonySpeech.listenSpeech())
		print("\n\nI heard: " + text)

                # Record the transcript with a timestamp
                with open(f"working/{current_mode}/recordings/{timestamp}_recording.txt", "w") as f:
                    f.write(text)

                playsound(ack, block=False)
                # playsound(ambient, block=False)

                load_characters_from_config(api)
                for channel in api.channels.keys():
                    api.process_text_and_synthesize(channel, text, timestamp, responding_prompt1)
                
                break  # If everything is successful, break out of the loop

    #        except sr.UnknownValueError:
    #            print("Google Speech Recognition could not understand audio")
    #            playsound(fail)
    #            retry_count += 1  # Increment the retry count

    #        except sr.RequestError as e:
    #            print("\n\nCould not request results from Google Speech Recognition service; {0}".format(e))
    #            playsound(problem)
    #            retry_count += 1  # Increment the retry count

        	
		if retry_count == MAX_RETRIES:
            		print("Max retries reached. Exiting.")
    
    chatter_finished_event.set()

# Class definitions

class Character:

    def __init__(self, name, description, system_message, default_channel, eleven_voice_id):
        self.name = name
        self.description = description
        self.system_message = system_message
        self.default_channel = default_channel
        self.eleven_voice_id = eleven_voice_id
        self.device_id, self.max_channels = get_device_with_max_channels()

class ChannelAPI:
    
    def __init__(self, current_mode=None):
        self.current_mode = current_mode
        self.channels = {}
        self.characters = {}
        self.sound_playback_lock = threading.Lock() # Create a lock for sequential sound playback
        self.audio_playback_complete = threading.Event() # Create a signal that all sound playback is complete

    def assign_character(self, character):
        if character.default_channel in self.channels:
            print(f"Channel {character.default_channel} already has a character assigned.")
            return False
        self.channels[character.default_channel] = character.name
        self.characters[character.default_channel] = character
        # print(f"\n\nAssigned {character.name} to channel {character.default_channel}.")
        return True
    
    def get_assigned_character(self, channel):
        return self.characters.get(channel, None)
    
    def process_text_and_synthesize(self, channel, text, timestamp, responding_prompt1):
        character = self.get_assigned_character(channel)
        if character is None:
            print(f"\n\nNo character assigned to channel {channel}.")
            return

        def worker(channel, text, character, timestamp, responding_prompt1):
            # Sending request to remote text processing API
            processed_text = self.send_to_language_model(text, character, timestamp, responding_prompt1)
            # Sending to remote audio synthesis API
            audio_data, audio_filename, final_text = self.send_to_audio_synthesis(channel, processed_text, timestamp)
            # Route the synthesized audio to the assigned channel
            # Use the lock to ensure sequential sound playback
            # with self.sound_playback_lock:
            self.route_to_channel(channel, text, character, audio_data, audio_filename, character.max_channels, character.device_id, final_text)

        self.start_threads_staggered(worker, channel, text, character, timestamp, responding_prompt1)

    def send_to_language_model(self, text, character, timestamp, responding_prompt1):

        # Come up with responses
        prompt = f"{responding_prompt1}\n\n'{text}.'"

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": f"{character.system_message}"},
                {"role": "user", "content": f"{prompt}"}
                ]
            )
        # Save the responses to a local file with an epoch timestamp
        filename = f"working/{self.current_mode}/audio/{timestamp}_{character.name}_responses.json"
        with open(filename, "w") as f:
            f.write(response.choices[0]['message']['content'])
        responses = response.choices[0]['message']['content']
        return responses

    def send_to_audio_synthesis(self, channel, processed_text, timestamp):

        character = self.get_assigned_character(channel)
        eleven_voice_id = character.eleven_voice_id

        if character is None:
            print(f"No character assigned to channel {channel}.")
            return

        # Read the text from the input file
        with open(f'working/{self.current_mode}/audio/{timestamp}_{character.name}_responses.json', "r") as f:
            text = f.read()

        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": eleven_labs_api_key
        }

        # Request TTS from remote API
        final_text = text
        response = requests.post(eleven_voice_id, json={"text": text, "voice_settings": {"stability": 0, "similarity_boost": 0}}, headers=headers)
        # print(f"ElevenLabs API response: {response.status_code}")
        if response.status_code:

            # Save remote TTS output to a local audio file with an epoch timestamp
            filename = f"working/{self.current_mode}/audio/{timestamp}_{character.name}_response.mp3"
            with open(filename, "wb") as f:
                f.write(response.content)
           
        return response.content, filename, final_text

    def start_threads_staggered(self, worker_func, channel, text, character, timestamp, responding_prompt1, delay=0):
        # Start the thread
        threading.Thread(target=worker_func, args=(channel, text, character, timestamp, responding_prompt1)).start()
        # Sleep for the specified delay time
        time.sleep(delay)
       
    def route_to_channel(self, channel, text, character, audio_data, filename, max_channels, max_device_id, final_text):
        character = self.get_assigned_character(channel)
        if character is None:
            print(f"No character assigned to channel {channel}.")
            return

        # Load audio file using pydub from disk
        # TODO: just read the audio data, it will save on IO
        sound = AudioSegment.from_file(filename, format="mp3")

        # Convert sound to numpy array and normalize
        samples = np.array(sound.get_array_of_samples()).astype(np.float32) / (2**15)
        # print(max_channels) #8
        # print(max_device_id) #0
        max_output_channels = sd.query_devices(max_device_id)['max_output_channels']

        if max_channels == 2:
            channel = 0

        if channel >= max_output_channels:
            raise ValueError(f"The device only has {max_channels} output channel(s).")

        # Create an empty array with the correct number of output channels
        zeros = np.zeros((len(samples), max_output_channels), dtype=np.float32)

        # Copy the mono audio data to the desired channel
        zeros[:, channel] = samples

        # Wait for previous sound to finish
        # self.audio_playback_complete.wait()

        # Lock sound playback
        with self.sound_playback_lock:
            # Reset the event indicating that sound playback has started
            self.audio_playback_complete.clear()

            # Stream the audio 
            print(f"\n\n{character.name} is speaking on {channel}:\n\n{final_text}")
            with sd.OutputStream(device=max_device_id, channels=max_output_channels, samplerate=sound.frame_rate) as stream:
                stream.write(zeros)

                # # TODO: Add a listener here to indicate Plantoid is speaking

            # # Reset the event for the next run
            self.audio_playback_complete.set()
            # playsound("global/media/cleanse.mp3", block=False)

        pass
