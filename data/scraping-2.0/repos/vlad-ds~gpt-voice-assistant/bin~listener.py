import argparse
import os
import re

from dotenv import load_dotenv
import sounddevice as sd
import numpy as np
import openai
from playsound import playsound
from pynput.keyboard import Controller as KeyboardController, Key, Listener
from rich.console import Console
from rich.markdown import Markdown
from scipy.io import wavfile
from termcolor import colored, cprint

from oa import apply_whisper, chatgpt
from polly import text_to_speech_oa

load_dotenv()
key_label = os.environ.get("RECORD_KEY", "ctrl_r")
RECORD_KEY = Key[key_label]
print(RECORD_KEY)

recording = False
audio_data = []
sample_rate = 16000
keyboard_controller = KeyboardController()
message_history = []


def process_response_for_audio(response: str) -> str:
    # substitute everything between ``` and ``` with a default string
    response = re.sub(r'```.*?```', '(See code in terminal)', response, flags=re.M|re.S)
    return response


def main(no_audio: bool = False):
    def on_press(key):
        global recording
        global audio_data
        # When the right shift key is pressed, start recording
        if key == RECORD_KEY:
            recording = True
            audio_data = []
            print("Recording started...")

    def on_release(key):
        global recording
        global audio_data
        global message_history
        console = Console()
        
        # When the right shift key is released, stop recording
        if key == RECORD_KEY:
            recording = False
            print("Recording stopped.")
            
            try:
                audio_data_np = np.concatenate(audio_data, axis=0)
            except ValueError as e:
                print(e)
                return
            
            audio_data_int16 = (audio_data_np * np.iinfo(np.int16).max).astype(np.int16)
            wavfile.write('recording.wav', sample_rate, audio_data_int16)

            transcript = None
            try:
                transcript = apply_whisper('recording.wav', 'transcribe')
            except openai.error.InvalidRequestError as e:
                print(e)
            
            if transcript:
                print(colored("User:", "red"), transcript)

                # clear history when "clear" is said
                letters_only = ''.join([char for char in transcript if char.isalpha()])
                if letters_only.lower().strip() == 'clear':
                    message_history = []
                    print(colored("Assistant:", "green"), colored("History cleared.", "red"))
                    playsound('bin/sounds/clear.mp3')
                    return
                
                history = chatgpt(transcript, message_history)
                message_history = history
                response = history[-1]['content']
                print(colored("Assistant:", "green"))
                console.print(Markdown(response))
                response_processed = process_response_for_audio(response)
                text_to_speech_oa(response_processed)
                if not no_audio:
                    playsound('output.mp3')

    def callback(indata, frames, time, status):
        if status:
            print(status)
        if recording:
            audio_data.append(indata.copy())  # make sure to copy the indata

    with Listener(on_press=on_press, on_release=on_release) as listener:
        # This is the stream callback
        with sd.InputStream(callback=callback, channels=1, samplerate=sample_rate):
            # Just keep the script running
            listener.join()


if __name__ == "__main__":
    # get --no-audio flag
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-audio", action="store_true")
    
    args = parser.parse_args()

    print(colored("Assistant:", "green"), "Started.")
    # playsound('bin/sounds/start.mp3')
    try:
        main(no_audio = args.no_audio)
    except KeyboardInterrupt:
        print(colored("Assistant:", "green"), "Closing.")
        # playsound('bin/sounds/stop.mp3')
