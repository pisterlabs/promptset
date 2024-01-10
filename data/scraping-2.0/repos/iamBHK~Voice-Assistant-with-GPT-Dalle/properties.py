# Python cross audio library to play and record the audio
import pyaudio
# # listen voice
import speech_recognition as speech
# # Python text to speech
import pyttsx3

import sys
import datetime
import json
import os

# For GPT4
import openai
# Get this from your open ai account
openai.organization = "***"
openai.api_key = '***'

listener = speech.Recognizer()
machine = pyttsx3.init()

voices = machine.getProperty('voices')
# Change to female voice
machine.setProperty('voice', voices[1].id)
# Slow down the speed rate of voice
machine.setProperty('rate',190)