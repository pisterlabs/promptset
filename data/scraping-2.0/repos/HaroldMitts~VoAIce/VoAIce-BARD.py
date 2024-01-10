print("\033c", end="")
print("VoAIce-BARD.py")
print("May 18, 2023")
print("This script assumes you have the voices.json file in the same directory as this script.")
print("This script assumes you have the api_keys.json file in the same directory as this script.")
print("This script assumes you have the GPT-4 API key in the api_keys.json file.")
print("This script assumes you have the Azure Speech API key in the api_keys.json file.")
print("This script assumes you have the Azure Speech region in the api_keys.json file.")
print("This script assumes you have the ffmpeg installed.")



import json
import os
import subprocess
import tempfile
import time
import azure.cognitiveservices.speech as speechsdk
import openai
import tkinter as tk
import tkinter.filedialog
import csv
import threading
from queue import Queue

from azure.cognitiveservices.speech import SpeechConfig, SpeechRecognizer
from tkinter import *

def clear_screen():
    os.system("cls") if os.name == "nt" else os.system("clear")

if __name__ == "__main__":
    clear_screen()



def load_api_keys(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def transcribe_audio(speech_config):
    recognizer = SpeechRecognizer(speech_config)
    with tempfile.NamedTemporaryFile() as f:
        recognizer.save_audio_to_file(f.name)
        audio_data = f.read()
    return recognizer.recognize_async(audio_data).result()


def generate_response(input_text, conversation_history):
    client = openai()
    response = client.create_response(
        engine="davinci",
        prompt=input_text,
        temperature=0.7,
        top_p=0.9,
        tokens=50,
        conversation=conversation_history,
    )
    return response.choices[0].text


def synthesize_and_save_speech(speech_config, response_text, file_path):
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)
    result = speech_synthesizer.speak_text_async(response_text).get()

    with open(file_path, "wb") as f:
        f.write(result.audio_data)


def play_audio(audio_file_path):
    subprocess.Popen(["ffmpeg", "-i", audio_file_path])


def remove_temp_files(file_path):
    os.remove(file_path)


class Application(Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.api_keys = load_api_keys("api_keys.json")
        self.speech_config = SpeechConfig(
            key=self.api_keys["azure"]["speech"], region="eastus"
        )
        self.openai_client = openai(client_id=self.api_keys["openai"]["client_id"])
        self.conversation_history = []
        self.is_running = False

        self.create_widgets()

    def create_widgets(self):
        self.text_input = Entry(self)
        self.send_button = Button(self, text="Send", command=self.send_message)
        self.conversation_display = Text(self, height=10, width=40)

        self.text_input.pack()
        self.send_button.pack()
        self.conversation_display.pack()

    def update_display(self):
        self.conversation_display.delete(1.0, END)
        for message in self.conversation_history:
            self.conversation_display.insert(END, message)

    def start(self):
        self.is_running = True
        while self.is_running:
            time.sleep(0.5)
            if not self.text_input.get():
                continue
            input_text = self.text_input.get()
            self.conversation_history.append(input_text)
            response_text = generate_response(input_text, self.conversation_history)
            synthesize_and_save_speech(self.speech_config, response_text, "audio.wav")
            play_audio("audio.wav")
            remove_temp_files("audio.wav")
            self.text_input.delete(0, END)

    def send_message(self):
        input_text = self.text_input.get()
        self.conversation_history.append(input_text)
        response_text = generate_response(input_text, self.conversation_history)
        synthesize_and_save_speech(self.speech_config, response_text, "audio.wav")
        play_audio("audio.wav")
        remove_temp_files("audio.wav")
        self.text_input.delete(0, END)


if __name__ == "__main__":
    root = Tk()
    application = Application(master=root)
    application.mainloop()
