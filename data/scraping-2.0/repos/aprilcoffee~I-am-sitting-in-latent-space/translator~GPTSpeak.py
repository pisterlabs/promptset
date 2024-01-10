import os
import random
from translate import Translator
import time
import openai
import json
import datetime

import argparse
import math

from pythonosc.dispatcher import Dispatcher
from pythonosc import osc_server
from pythonosc import udp_client
import requests

url = "http://172.20.10.3:5002/api/tts?text="
import config 
openai.api_key = config.openai_api_key
#from playsound import playsound

import os, glob
questions = [
"Start the dance movement",
"How do musicians motivate the dancers to start the play",                  
"how can a human embody artificial intelligence?", 
"who are some of your political role models and why", 
"why is the system so delightfully toxic", 
"why is the female breast sexualised through mark zuckerberg's censorship on instagram?"
]

answers = [
"Thirty seconds of free-style movement, allowing yourself to be deeply immersed in the music,  and taking inspiration from your fellow dancers and musicians as you weave fluid and energetic movements together with sharp, angular accents.",
"Use your body to convey the energy and urgency of artificial intelligence, while the violinist builds tension in the background and the bass and electronics create a pulsating beat, until your movements signal the dancers to join you and start the performance.",
"Dancers, convey the fluid yet precise movements of technology while the violin and bass players express the harmony and discord of machine and human through their respective instruments, creating a synthetic blend of audio and movement.",
"Perform a slow and smooth solo combining fluid footwork, gestural movements, and delicate bow strokes on the violin to reflect deep contemplation as you think of your political role models and their virtues.",
"Imitate the gradual release of toxicity within your bodies through bold and experimental kinetic dance movements, complimented by eerily coordinated melodies of disjointed glitch electronica to produce a world of atmospheric splendor.",
"Using subtle seductive gesturing, imaginatively portray society enforcing unequal gender expectations based on controversial morality disapproved-categorized otherwise liberal may engage one's perception while eye-arousing what does and will remain visible.."
]


def sendOSCtoQuestion(output):
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="192.168.0.189",
    help="The ip of the OSC server")
    parser.add_argument("--port", type=int, default=5555,
    help="The port the OSC server is listening on")
    args = parser.parse_args()

    client = udp_client.SimpleUDPClient(args.ip, args.port)
    client.send_message("/question", "question:" + output)

def sendOSCtoAnswer(output):
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="192.168.0.189",
    help="The ip of the OSC server")
    parser.add_argument("--port", type=int, default=5555,
    help="The port the OSC server is listening on")
    args = parser.parse_args()

    client = udp_client.SimpleUDPClient(args.ip, args.port)
    client.send_message("/answer", "answer:" + output)


def voice_handler(unused_addr,args,volume):
    #print("go for record audio mode")
    print( volume)
    #number = volume 


    input_text = questions[volume-1]
    print(input_text)
    sendOSCtoQuestion(input_text)

    time.sleep(5)
    output_text = answers[volume-1]
    print(output_text)
    sendOSCtoAnswer(output_text)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--ip",
    default="127.0.0.1", help="The ip to listen on")
    parser.add_argument("--port",
    type=int, default=5005, help="The port to listen on")
    args = parser.parse_args()

    dispatcher = Dispatcher()
    dispatcher.map("/voice", voice_handler, "voice")
    server = osc_server.ThreadingOSCUDPServer(
    (args.ip, args.port), dispatcher)
    print("Serving on {}".format(server.server_address))

    server.serve_forever()