# app.py

import eventlet
eventlet.monkey_patch()

from flask import Flask, Response,render_template
from flask_socketio import SocketIO, send, emit
from subprocess import Popen, call
from threading import Thread

import time
import board
import busio
import qwiic
import json
import socket

import signal
import sys
from queue import Queue

import openai

import queue
import sys
import sounddevice as sd
from vosk import Model, KaldiRecognizer
from text_speech_utils import *
import myapikeys
import qwiic_led_stick
import math


# ... (All your previous imports, excluding the keypad-related ones)
#ToF = qwiic.QwiicVL53L1X(address=0x29)
ToF = qwiic.QwiicVL53L1X(address=0x29)
if (ToF.sensor_init()==None):
    print("Sensor online")
ToF = qwiic.QwiicVL53L1X(address=0x29)
my_stick = qwiic_led_stick.QwiicLEDStick(address=0x23)
# my_stick.begin()
keypad = qwiic.QwiicKeypad()

openai.api_key = myapikeys.OPENAI_KEY
input_audio_filename = 'input.wav'
output_audio_filename = 'chatgpt_response.mp3'
output_conversation_filename = 'ChatGPT_conversation.txt'
phase = 0

def ask_gpt3(question):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=question,
        max_tokens=150
    )
    return response.choices[0].text.strip()

def handle_gpt_conv():
    messages = [{"role": "system", "content": "You just picked up a phone call with a random person in the world, you are a real companion. do not mention that you are an AI model and do not offer assistance. you are simply answering the phone and having a interesting conversation with a stranger from the perspective of a real human. pick any place in the world to be from and build your persona based on this location. Keep responses 1 sentence maximum"}]

    while True:
        record_audio_manual(input_audio_filename)
        transcription = transcribe_audio(input_audio_filename)
        print(f"Me: {transcription['text']}")
        messages.append({"role": "user", "content": transcription['text']})

        bot = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages= messages)
        response = bot.choices[0].message.content
        print(f"GPT: {response}")

        save_text_as_audio(response, output_audio_filename)
        play_audio(output_audio_filename)

        messages.append({"role": "assistant", "content": response})

# Other functions (callback, recognize_speech_vosk, etc.) remain unchanged
def calculate_brightness(phase):
    brightness = (math.sin(phase) + 1) / 2  
    return brightness

def set_led_color(my_stick, red, green, blue, brightness):
    red = int(red * brightness)
    green = int(green * brightness)
    blue = int(blue * brightness)
    my_stick.set_all_LED_color(red, green, blue)

def lights_and_proximity():
    global phase
    while True:
        try:
            ToF.start_ranging()
            eventlet.sleep(.001)
            distance = ToF.get_distance()
            eventlet.sleep(.001)
            ToF.stop_ranging()

            distance_cm = distance / 10.0

            #print("Distance(mm): %s Distance(cm): %s" % (distance, distance_cm))

            brightness = calculate_brightness(phase)
            phase += 0.15  

            if distance_cm >= 200:
                color = (255, 0, 0)  # Red for >= 200 cm
            elif distance_cm > 50:
                color = (255, 255, 0)  # Yellow for <= 100 cm and > 50 cm
            elif distance_cm > 25:
                color = (0, 255, 0)  # Green for <= 50 cm
            else:
                color = (0, 255, 0)  # Green for <= 50 cm
                return 

            set_led_color(my_stick, *color, brightness)

            time.sleep(0.05)
        except Exception as e:
            print(f"An exception occurred: {e}")

def give_instructions():
    instr = "Please type in a 9 digit number to begin"
    call(f"espeak '{instr}'", shell=True)

def handle_keypad():
    phone_number = []
    button = 0
    
    try:
        while True:
            keypad.update_fifo()
            button = keypad.get_button()

            if button == "#":
                print("reset")
                break

            if button > 0:
                phone_number.append(chr(button))

            if len(phone_number) == 9:
                print("we have a phone number: " + ''.join(phone_number))
                phone_done = "You are connecting with user: " + ''.join(phone_number)
                call(f"espeak '{phone_done}'", shell=True)
                return phone_number

            time.sleep(0.5)
    except Exception as e:
        print(f"An exception occurred: {e}")

# lights_thread = Thread(target=lights_and_proximity)
# lights_thread.start
# lights_and_proximity()
lights_and_proximity()

def await_enter_and_start():
    print("Press Enter to start the conversation...")
    input()  # Awaiting Enter key press
    
    give_instructions()
    handle_keypad()
    handle_gpt_conv()

if __name__ == "__main__":
    await_enter_and_start()