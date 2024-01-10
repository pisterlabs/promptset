#!/usr/bin/python3

import openai
from gtts import gTTS
import os
import time
import wave
#import api_secrets
import assemblyai as aai
import datetime
import requests
import string
import csv
import boto3
import serial
import speech_recognition as sr
import threading

class Verbal:
	def __init__(self):
		pass
		
	def textToSpeech(text):
		speech = gTTS(text=text, lang="en", slow=False, tld="ie")
		speech.save("textToSpeech.mp3")

		audio = "textToSpeech.mp3"
		os.system("mpg321 -q " + audio)

	def transcribeAudioFile(self, audioFileName):
		recognizer = sr.Recognizer()

		# Load the audio file using the speech_recognition library
		with sr.AudioFile(audioFileName) as source:
			audio_data = recognizer.record(source)

		# Use the recognizer to transcribe the audio data
		try:
			transcription = recognizer.recognize_google(audio_data, show_all=False)
			return transcription
		except sr.UnknownValueError:
			print("Google Speech Recognition could not understand audio")
			return None
		except sr.RequestError as e:
			print(f"Could not request results from Google Speech Recognition service; {e}")
			return None

			
	def speechToText(audioFileName, recTime):
		loopTime = recTime*4
		rate = 16000
		print("Start speaking:")
		os.system("arecord -q -D plughw:2,0 -d " + str(recTime) + " -r" + str(rate) + " " + audioFileName)
		print("Processing audio")

		#transcribe audio
		AUDIO_FILE_URL = audioFileName
		
		recognizer = sr.Recognizer()

		# Load the audio file using the speech_recognition library
		with sr.AudioFile(audioFileName) as source:
			audio_data = recognizer.record(source)

		# Use the recognizer to transcribe the audio data
		try:
			transcription = recognizer.recognize_google(audio_data, show_all=False)
			request = transcription
		except sr.UnknownValueError:
			print("Google Speech Recognition could not understand audio")
			request = None
		except sr.RequestError as e:
			print(f"Could not request results from Google Speech Recognition service; {e}")
			request = None
			
		return request
		
