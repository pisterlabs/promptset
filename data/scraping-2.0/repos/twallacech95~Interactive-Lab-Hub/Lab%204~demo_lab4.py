# app.py

import eventlet
eventlet.monkey_patch()

from flask import Flask, Response,render_template
from flask_socketio import SocketIO, send, emit
from subprocess import Popen, call

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
from threading import Thread
from elevenlabs import set_api_key, generate, play
import requests


set_api_key(myapikeys.ELEVENLABS_KEY)

openai.api_key = myapikeys.OPENAI_KEY
input_audio_filename = 'input.wav'
output_audio_filename = 'chatgpt_response.mp3'
output_conversation_filename = 'ChatGPT_conversation.txt'

def calculate_brightness(phase):
    # Create a pulsing effect by varying brightness with a sine wave
    brightness = (math.sin(phase) + 1) / 2  # Convert range from -1 -> 1 to 0 -> 1
    return brightness  # Keep brightness as a float between 0 and 1

def set_led_color(my_stick, red, green, blue, brightness):
    # Scale RGB values based on the brightness
    red = int(red * brightness)
    green = int(green * brightness)
    blue = int(blue * brightness)
    # Set the color for all LEDs
    my_stick.set_all_LED_color(red, green, blue)

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


ToF = qwiic.QwiicVL53L1X(address=0x29)
if (ToF.sensor_init() == None):   # Begin returns 0 on a good init
    print("Sensor online!\n")

my_stick = qwiic_led_stick.QwiicLEDStick(address=0x23)


phase = 0  # Initial phase for sine wave
keypad = qwiic.QwiicKeypad(address=0x52)

#### ELIMINATE THIS

# i2c = busio.I2C(board.SCL, board.SDA)
# keypad = qwiic.get_devices()[9]
# every = qwiic.get_devices()
# print(every)

# hostname = socket.gethostname()
# hardware = 'plughw:2,0'

# app = Flask(__name__)
# socketio = SocketIO(app)
# q = queue.Queue()

def lights_and_proximity():
    global phase
    while True:
        try:
            ToF.start_ranging()
            eventlet.sleep(.001)
            distance = ToF.get_distance()
            eventlet.sleep(.001)
            ToF.stop_ranging()

            distance_cm = distance / 10.0  # Convert mm to cm

            #print("Distance(mm): %s Distance(cm): %s" % (distance, distance_cm))

            brightness = calculate_brightness(phase)
            phase += 0.15  # Faster phase increment for a quicker pulse

            # Change LED color based on distance
            if distance_cm >= 200:
                color = (255, 0, 0)  # Red for >= 200 cm
            elif distance_cm > 50:
                color = (255, 255, 0)  # Yellow for <= 100 cm and > 50 cm
            else:
                color = (0, 255, 0)  # Green for <= 50 cm

            set_led_color(my_stick, *color, brightness)

            time.sleep(0.05)  # Shorter delay for quicker updates and smoother transition

        except Exception as e:
            print(e)


def callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))

def recognize_speech_vosk():
    model = Model(lang="en-us")
    rec = KaldiRecognizer(model, 16000)

    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16', channels=1, callback=callback):
        print("Listening, press Ctrl+C to exit...")
        while True:
            data = q.get()
            if rec.AcceptWaveform(data):
                result = rec.Result()
                result_json = json.loads(result)
                recognized_text = result_json.get("text", "")
                return recognized_text

def handle_speak(val):
    call(f"espeak '{val}'", shell=True)


def give_instructions():
    instr = "Please type in a 9 digit number to begin"
    call(f"espeak '{instr}'", shell=True)


def test_connect():
    print('connected')
    emit('after connect',  {'data':'Lets dance'})

def handle_message(val):
    emit('pong-gps', sox.acceleration) 

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
                while True:
                    handle_gpt_conv()

            time.sleep(0.5)
    except Exception as e:
        print(f"An exception occurred: {e}")

lights_thread = Thread(target=lights_and_proximity)
lights_thread.start()


if keypad.begin() == False:
    print('The Qwicc Keypad is not connected', file=sys.stderr)
    
else:
    print('The Qwiic Keypad is connected.')
    print("Please type in a 9 digit phone number.")
    give_instructions()
    handle_keypad()




