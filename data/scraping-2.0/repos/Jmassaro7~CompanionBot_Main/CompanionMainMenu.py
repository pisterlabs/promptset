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
import RPi.GPIO as GPIO
import threading
import RPi.GPIO as GPIO

from importlib import import_module

from TimeUtility import TimeUtility
from Verbal import Verbal
from Weather import Weather
from Emotion import Emotion
from Arduino import Arduino
import api_secrets


openai.api_key = api_secrets.API_KEY_OPENAI_CHATGPT
#api_keys.API_KEY_OPENAI_CHATGPT

timeUtility = TimeUtility()
verbal = Verbal()
weather = Weather()
emotion = Emotion()
arduino = Arduino()

is_active = 16

GPIO.setmode(GPIO.BOARD)
GPIO.setup(is_active,GPIO.IN)

Arduino.initialize()

limit = "30"                # Word limit for ChatGPT response
name = "User"              # Name that you would like ChatGPT to address you as
language = "en"             # Language of ChatGPT response
communicate = False         # Communicate toggle


while(True):
    if(GPIO.input(is_active) == False):
        print("Button pressed")
        # Initial Prompt: Ask user if they would like to talk
        text = "Hello " + name + "! Would you like to talk to me? "
        displayThread = threading.Thread(target=Emotion.display_animations, args=("happy",), daemon=True)
        displayThread.start()
        Verbal.textToSpeech(text)
        displayThread = threading.Thread(target=Emotion.display_animations, args=("listeningThree",), daemon=True)
        displayThread.start()
        request = Verbal.speechToText("initialPrompt.wav", 3)

        if request is None:
            print("Request is None")
            request = ""

        # Input prompt to take user input in response to Initial Prompt

        if "yes" in request.lower():     # If "yes" is contained in the response
            communicate = True          # Set communicate toggle to True

        if communicate == True:
            displayThread = threading.Thread(target=Emotion.display_animations, args=("happy",), daemon=True)
            displayThread.start()
            Verbal.textToSpeech("Cool! Welcome to my main menu. What would you like to do?")

        while communicate == True:
            if(GPIO.input(is_active) == False):
                break
            #changed time for choice to 6 seconds
            displayThread = threading.Thread(target=Emotion.display_animations, args=("listeningThree",), daemon=True)
            displayThread.start()
            request = Verbal.speechToText("choicePrompt.wav", 4)

            if request is None:
                print("Request is None")
                displayThread = threading.Thread(target=Emotion.display_animations, args=("confused",), daemon=True)
                displayThread.start()
                Verbal.textToSpeech("I'm sorry, I did not catch that. Could you please repeat yourself?")
                request = ""

            if "wait" in request.lower():
                displayThread = threading.Thread(target=Emotion.display_animations, args=("calm",), daemon=True)
                displayThread.start()
                Verbal.textToSpeech("No worries. I'll wait for you. If you'd like me to start listening again, say 'wake up'.")
                while "wake" not in request.lower() and "up" not in request.lower():
                    os.system("aplay idling.wav")
                    displayThread = threading.Thread(target=Emotion.display_animations, args=("calm",), daemon=True)
                    displayThread.start()
                    request = Verbal.speechToText("waitPhrase.wav", 5)
                    if request is None:
                        print("Request is None")
                        request = ""
                    print("Take your time. I will wait for you.")
                displayThread = threading.Thread(target=Emotion.display_animations, args=("happy",), daemon=True)
                displayThread.start()
                Verbal.textToSpeech("That was a nice little nap! What else would you like to do from my main menu?")
            
            if "thank you" in request.lower() or "no" == request.lower() or "thanks" in request.lower():
                    communicate = False
                    print(communicate)
                    break
            
            
            if "chat" in request.lower() or "talk" in request.lower():
                displayThread = threading.Thread(target=Emotion.display_animations, args=("happy",), daemon=True)
                displayThread.start()
                Verbal.textToSpeech("Sure thing! What's up?")
                
                while True:
                    displayThread = threading.Thread(target=Emotion.display_animations, args=("listeningSeven",), daemon=True)
                    displayThread.start()
                    chatgptRequest = Verbal.speechToText("chatgptPrompt.wav", 5)
                    
                    if chatgptRequest is None:
                        print("chatgptRequest is None")
                        displayThread = threading.Thread(target=Emotion.display_animations, args=("confused",), daemon=True)
                        displayThread.start()
                        Verbal.textToSpeech("I'm sorry, I did not catch that. Could you please repeat yourself?")
                        chatgptRequest = ""
                        continue
                    
                    if "wait" in chatgptRequest.lower():
                        displayThread = threading.Thread(target=Emotion.display_animations, args=("happy",), daemon=True)
                        displayThread.start()
                        Verbal.textToSpeech("No worries. I'll wait for you. If you'd like me to start listening again, say 'wake up'.")
                        while "wake" not in chatgptRequest.lower() and "up" not in chatgptRequest.lower():
                            os.system("aplay idling.wav")
                            displayThread = threading.Thread(target=Emotion.display_animations, args=("calm",), daemon=True)
                            displayThread.start()
                            chatgptRequest = Verbal.speechToText("waitPhrase.wav", 5)
                            print("Take your time. I will wait for you.")
                            
                            if chatgptRequest is None:
                                print("timerInput is None")
                                displayThread = threading.Thread(target=Emotion.display_animations, args=("confused",), daemon=True)
                                displayThread.start()
                                chatgptRequest = ""
                        displayThread = threading.Thread(target=Emotion.display_animations, args=("happy",), daemon=True)
                        displayThread.start()
                        Verbal.textToSpeech("That was a nice little nap! What else would you like to chat about?")
                        displayThread = threading.Thread(target=Emotion.display_animations, args=("listeningSeven",), daemon=True)
                        displayThread.start()
                        chatgptRequest = Verbal.speechToText("chatgptPrompt.wav", 5)
                        if chatgptRequest is None:
                            print("Request is None")
                            displayThread = threading.Thread(target=Emotion.display_animations, args=("confused",), daemon=True)
                            displayThread.start()
                            chatgptRequest = ""
                    
                    if "thank you" in chatgptRequest.lower() or "no" == chatgptRequest.lower() or "thanks" in chatgptRequest.lower():
                        break
                    displayThread = threading.Thread(target=Emotion.display_animations, args=("thinking",), daemon=True)
                    displayThread.start()
                    completion = openai.ChatCompletion.create(model="gpt-4-1106-preview", messages=[{"role": "user", "content": chatgptRequest + ". Keep response under " + limit + " words."}])
                    print(completion.choices[0].message.content)

                    Verbal.textToSpeech(completion.choices[0].message.content)
                    Verbal.textToSpeech("What else would you like to chat about?")
                displayThread = threading.Thread(target=Emotion.display_animations, args=("happy",), daemon=True)
                displayThread.start()
                Verbal.textToSpeech("I enjoyed chatting with you. What else can I do for you from my main menu?")

            if "time " in request.lower() or "time." in request.lower() or "time" in request.lower() and "timer" not in request.lower():
                displayThread = threading.Thread(target=Emotion.display_animations, args=("surprised",), daemon=True)
                displayThread.start()
                
                #Call getTime() function from Time class
                current_time_str = TimeUtility.getTime()
                displayThread = threading.Thread(target=Emotion.display_animations, args=("surprised",), daemon=True)
                displayThread.start()
                Verbal.textToSpeech(current_time_str)
                displayThread = threading.Thread(target=Emotion.display_animations, args=("happy",), daemon=True)
                displayThread.start()
                Verbal.textToSpeech("Would you like to try something else from my main menu?")

            if "timer" in request.lower():
                timeUtility.set_timer()

            if("weather") in request.lower():
                while(True):
                    #displayThread = threading.Thread(target=Emotion.display_animations, args=("happy",), daemon=True)
                    #displayThread.start()
                    #Verbal.textToSpeech("Which city would you like to check the weather for?")
                    #displayThread = threading.Thread(target=Emotion.display_animations, args=("listeningThree",), daemon=True)
                    #displayThread.start()
                    #city = Verbal.speechToText("getWeather.wav", 3)
                    city = "Flint"
                    if city is None:
                        print("Request is None")
                        displayThread = threading.Thread(target=Emotion.display_animations, args=("confused",), daemon=True)
                        displayThread.start()
                        Verbal.textToSpeech("I did not hear the name of a city. Could you please try that again?")
                    if city is not None:
                        weatherData = weather.getWeather(city)
                        displayThread = threading.Thread(target=Emotion.display_animations, args=("confused",), daemon=True)
                        displayThread.start()
                        Verbal.textToSpeech(weatherData)
                        displayThread = threading.Thread(target=Emotion.display_animations, args=("happy",), daemon=True)
                        displayThread.start()
                        Verbal.textToSpeech("How else may I assist you?")
                        break

            if("temperature") in request.lower():
                while(True):
                    #displayThread = threading.Thread(target=Emotion.display_animations, args=("happy",), daemon=True)
                    #displayThread.start()
                    #Verbal.textToSpeech("Which city would you like to check the temperature for?")
                    
                    #displayThread = threading.Thread(target=Emotion.display_animations, args=("listeningThree",), daemon=True)
                    #displayThread.start()
                    #city = Verbal.speechToText("getWeather.wav", 3)
                    city = "Flint"
                    if city is None:
                        print("Request is None")
                        displayThread = threading.Thread(target=Emotion.display_animations, args=("confused",), daemon=True)
                        displayThread.start()
                        Verbal.textToSpeech("I did not hear the name of a city. Could you please try that again?")
                    if city is not None:
                        temperature = weather.getTemperature(city)
                        displayThread = threading.Thread(target=Emotion.display_animations, args=("confused",), daemon=True)
                        displayThread.start()
                        Verbal.textToSpeech(temperature)
                        displayThread = threading.Thread(target=Emotion.display_animations, args=("happy",), daemon=True)
                        displayThread.start()
                        Verbal.textToSpeech("Can I assist you in any other way?")
                        break
                
            if "emotions" in request.lower() or "emotion" in request.lower():
                displayThread = threading.Thread(target=Emotion.display_animations, args=("confused",), daemon=True)
                displayThread.start()
                Verbal.textToSpeech("Please look into the camera and maintain your expression! A picture will be taken in 3, 2, 1")
                os.system("libcamera-still -n -o test.jpg")
                response = Emotion.evaluateEmotions()
                displayThread = threading.Thread(target=Emotion.display_animations, args=("confused",), daemon=True)
                displayThread.start()
                if(response is not None):
                    Verbal.textToSpeech(response)
                else:
                    Verbal.textToSpeech("I could not detect a face")
                displayThread = threading.Thread(target=Emotion.display_animations, args=("happy",), daemon=True)
                displayThread.start()
                Verbal.textToSpeech("Can I help you in any other way?")
                
            if("dance") in request.lower():
                displayThread = threading.Thread(target=Emotion.display_animations, args=("dance",), daemon=True)
                displayThread.start()
                Verbal.textToSpeech("Here's a little jig")
                displayThread = threading.Thread(target=Emotion.display_animations, args=("disgusted",), daemon=True)
                displayThread.start()
                Arduino.arduinoCall(1)
                displayThread = threading.Thread(target=Emotion.display_animations, args=("disgusted",), daemon=True)
                displayThread.start()
                Verbal.textToSpeech("I'm not a dancer. What else can I do for you from my main menu?")
            
            if ("drive") in request.lower():
                displayThread = threading.Thread(target=Emotion.display_animations, args=("happy",), daemon=True)
                displayThread.start()
                Verbal.textToSpeech("Suit up " + name + "! It's time to use the dabble app to take me for a spin!")

                Arduino.arduinoCall(2)
                displayThread = threading.Thread(target=Emotion.display_animations, args=("happy",), daemon=True)
                displayThread.start()
                Verbal.textToSpeech("That was fun! What else can I do for you from my main menu?")
             
        if "thank you" in request.lower() or "thanks" in request.lower():
            displayThread = threading.Thread(target=Emotion.display_animations, args=("happy",), daemon=True)
            displayThread.start()
            Verbal.textToSpeech("You're welcome " + name + ". Have a great day and remember that I am here for you if you ever need me!")

        else:
            displayThread = threading.Thread(target=Emotion.display_animations, args=("happy",), daemon=True)
            displayThread.start()
            Verbal.textToSpeech("No problem " + name + ". Have a great day and remember that I am here for you if you ever need me!")
    elif(GPIO.input(is_active) == True):
        print("Button not pressed")
