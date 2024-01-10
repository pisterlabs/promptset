import os
from openai import OpenAI
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from pynput import keyboard
import pyperclip
from dotenv import load_dotenv

class AudioTranscriber:
    def __init__(self):
        load_dotenv('./deck-dictation.env')
        self.api_key = os.environ["DECK_DICTATION_OPENAI_API_KEY"]
        self.client = OpenAI(api_key=self.api_key)
        self.audio_file_path = "speech.wav"
        self.fs = 44100  # Sample rate
        self.silence_threshold = 0.5  # Threshold for silence detection
        self.myrecording = []  # Start with an empty list
        self.is_silent = []  # List to keep track of the last two segments
        self.segment_length = 150  # Length of each segment in milliseconds
        self.number_of_silent_segments = 16  # Number of silent segments to stop after
        self.file_name = "speech.wav"

    def callback(self, indata, frames, time, status):
        volume_norm = np.linalg.norm(indata)
        if volume_norm < self.silence_threshold:
            print(".", end='', flush=True)  
            self.is_silent.append(True)
        else:
            print("|", end='', flush=True) 
            self.myrecording.append(indata.copy())
            self.is_silent.append(False)
        self.is_silent = self.is_silent[-self.number_of_silent_segments:]  

    def record_audio(self):
        blocksize = int(self.fs * self.segment_length / 1000)  

        with sd.InputStream(callback=self.callback, channels=1, samplerate=self.fs, blocksize=blocksize):
            print("Recording audio...")
            while True:
                if len(self.is_silent) == self.number_of_silent_segments and all(self.is_silent):
                    break

        print("Audio recording complete")
        self.myrecording = np.concatenate(self.myrecording) 
        write(self.file_name, self.fs, self.myrecording) 

    def transcribe(self):
        with open(self.audio_file_path, "rb") as audio_file:
            transcript = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )

        os.remove(self.audio_file_path)

        return(transcript.text)

class Hotkey_Handler:
    def __init__(self):
        self.combinations = [
            {keyboard.Key.space, keyboard.Key.ctrl, keyboard.KeyCode(char='l')},
            {keyboard.Key.space, keyboard.Key.ctrl, keyboard.KeyCode(char='L')}
        ]
        self.current = set()

    def on_press(self, key):
        if any([key in combination for combination in self.combinations]):
            self.current.add(key)
            if any(all(k in self.current for k in combination) for combination in self.combinations):
                audio_transcriber = AudioTranscriber()
                audio_transcriber.record_audio()
                transcription = audio_transcriber.transcribe()
                print(transcription)
                pyperclip.copy(transcription)

    def on_release(self, key):
        if any([key in combination for combination in self.combinations]):
            if key in self.current:
                self.current.remove(key)

    def listen(self):
        with keyboard.Listener(on_press=self.on_press, on_release=self.on_release) as listener:
            listener.join()



hotkey = Hotkey_Handler()
hotkey.listen()
