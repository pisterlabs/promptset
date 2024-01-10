import pandas as pd
import pyttsx3
import datetime
import speech_recognition as sr
from DEXI import JarvisAssistant
from DEXI.features.gui import Ui_MainWindow
import pyaudio
import smtplib
import geocoder
import wikipedia
# from newsapi import NewsApiClient
import sys
import newsapi
import pywhatkit
import requests
# from loginCREDS import senderMAIL, pwd, to
import pyautogui
import webbrowser as wb
import clipboard
import os
import time as tt
import pyjokes
import string
import subprocess as sp
import random
from time import sleep
import openai
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import QTimer, QTime, QDate, Qt
from PyQt5.QtGui import QMovie
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUiType
# import pygame
# import serial
# ser = serial.Serial('COM3', 9600)
obj = JarvisAssistant()
df = pd.read_csv('Active Faculty Data.csv')


engine = pyttsx3.init()     # Engine property modifications
engine.setProperty('rate', 235)
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

# def speak(text):
# voice = 'en-US-SteffanNeural'
# data = f'edge-tts --voice "{voice}" --text "{text}" --write-media data.mp3'
# os.system(data)

# pygame.init()
# pygame.mixer.init()
# pygame.mixer.music.load("data.mp3")

# try:
#     pygame.mixer.music.play()

#     while pygame.mixer.music.get_busy():
#         pygame.time.Clock().tick(10)

# except Exception as e :
#     print(e)

# finally:
#     pygame.mixer.music.stop()
#     pygame.mixer.quit()


def speak(text):        # Function to Speak
    engine.say(text)
    engine.runAndWait()
