
# import necessary packages
import speech_recognition as sr # https://pypi.org/project/SpeechRecognition/
import queue
from flask import Flask, render_template, Response
import time
import threading
import sys
import openai
import config
    
# 1ST THREAD - This is called from the background thread
# takes data from mic and adds directly to audio queue
def record_callback(_, audio:sr.AudioData):
    if config.toggle_variable:
        # only add data to audio queue if button has been pressed
        data = audio.get_raw_data()
        config.audio_queue.put_nowait(data)

# SETUP MICROPHONE RECORDING
# Reference: https://github.com/Uberi/speech_recognition/blob/master/examples/background_listening.py
# General Reference: https://github.com/Uberi/speech_recognition/blob/master/reference/library-reference.rst
def init_recording(pause_threshold = .5, energy = 300, dynamic_energy = False):
    # Init mic and recorder
    # Recorder will use the mic and callback function
    # to constantly listen for audio and add it to the audio_queue
    print('Detected Microphones:')
    print(sr.Microphone.list_microphone_names()) # list microphones
    mic = sr.Microphone()
    recorder = sr.Recognizer()
    config.sample_rate = mic.SAMPLE_RATE # use default sample rate for mic, whisper API can handle it

    # Represents the energy level threshold for sounds. Values below this threshold are considered silence, and values above this threshold are considered speech
    recorder.energy_threshold = energy

    # Represents the minimum length of silence (in seconds) that will register as the end of a phrase
    # Smaller values result in the recognition completing more quickly, but might result in slower speakers being cut off
    recorder.pause_threshold  = pause_threshold

    # Automatically increase/decrease energy to account for ambient noise
    recorder.dynamic_energy_threshold = dynamic_energy

    # Adjusts the energy threshold dynamically
    with mic as source:
        recorder.adjust_for_ambient_noise(source)

    # Start listening in another thread
    # Spawns a thread to repeatedly record phrases from mic
    # phrase time limit - maximum length of recorded phrases (seconds)
    recorder.listen_in_background(mic, callback=record_callback, phrase_time_limit=10)
    print("Microphone ready!")
    
    return mic, recorder
