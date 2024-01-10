import os
import importlib

def simple_bool(message):
    choose = input(message+" (y/n): ").lower()
    your_bool = choose in ["y", "yes"]
    return your_bool

def check_and_install_module(module_name):
    try:
        # Check if the module is already installed
        importlib.import_module(module_name)
    except ImportError:
        # If the module is not installed, try installing it
        install = simple_bool(
            "\n" + module_name + "  module is not installed.\nWould you like to install it?")
        if install:
            import subprocess
            subprocess.check_call(["pip", "install", module_name])
            print(f"The module '{module_name}' was installed correctly.")
        else:
            exit()

check_and_install_module('pyaudio')
check_and_install_module('keyboard')
check_and_install_module('pyperclip')
check_and_install_module('openai')
check_and_install_module('pyautogui')
check_and_install_module('wave')

# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
import openai

current_dir = os.getcwd()
api_key = None
if not os.path.isfile(current_dir + '/openai_api_key.txt'):
    with open(current_dir + '/openai_api_key.txt', 'w') as file:
        file.write(input('insert here your openai api key:'))

api_key = open(current_dir + '/openai_api_key.txt', 'r').read()
openai.api_key = str(api_key)

#----------------------------------------------

import time
import pandas as pd
import pyperclip
import pyautogui
import keyboard
import pyaudio
import wave
import time

input_device_id = 0
audio = pyaudio.PyAudio()

list = []
for index in range(audio.get_device_count()):
    info = audio.get_device_info_by_index(index)
    list.append(f"Device {index}: {info['name']}")
mics = pd.DataFrame(list)
input_device_id = input("/Select your microphone from the following list:\n"+mics.to_string(index=False))

chunk = 1024  # Number of frames per buffer
sample_format = pyaudio.paInt16  # 16 bits per sample
channels = 1  # Mono audio
rate = 44100  # Sampling rate in Hz


print("\nTo start record press Alt+A")
while True:
    
    if keyboard.is_pressed('Alt+A'):
        stream = audio.open(format=sample_format,
                            channels=channels,
                            rate=rate,
                            frames_per_buffer=chunk,
                            input=True,
                            input_device_index=int(input_device_id))

        frames = []
        print("Recording...")
        print("press alt+S to stop")
        

        while True:
            if keyboard.is_pressed('alt+S'):  # if key 'ctrl + c' is pressed
               break  # finish the loop
            else:
                data = stream.read(chunk)
                frames.append(data)

        print("Finished recording.")


        # Save the audio data to a WAV file
        filename = "recorded_audio.wav"
        wf = wave.open(filename, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(sample_format))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))
        wf.close()

        audio_file= open("recorded_audio.wav", "rb")
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
        text_value = transcript['text']
        pyperclip.copy(text_value)
        #pyperclip.paste()
        pyautogui.hotkey('ctrl', 'v')
        print('\n',text_value,'\n')
        print("\nTo start record press Alt+A")
