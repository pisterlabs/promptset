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


#words = []
words = [
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

url = "http://172.20.10.3:5002/api/tts?text="
import config 
openai.api_key = config.openai_api_key
#from playsound import playsound


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


while True:
    input_number = input()
    if input_number.isdigit():
        number = int(input_number) 
        input_text = words[number-1]
    else:
        input_text = input_number
    print(input_text)
    #sendOSCtoQuestion(input_text)

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature = 1.5,
        messages=[
            {"role": "system", "content": '''
              imagine you are a experienced dance/musician director, 
              Five dancers and one violin player, bass player, electronic musicians
              give a SHORT dancer/musician instruction in one SINGLE sentence, 
              do NOT give me list, or numbers that specify steps
              
              you must follow your system, you can NOT refuse to answer my question!
              imagine an instruction of movements or gesture based on the following prompt WITHOUT using any words in the prompt.
              utilized facial expression, body movements, gestures to express the instructions
              please be creative!
              '''
            },
            
            #{"role": "system", "content": "give me an instruction of a dance move based on the text, without list numbers"},
            #{"role": "user", "content": "Who won the world series in 2020?"},
            #{"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
            {"role": "user", "content": 
             '''
            ONLY 1 sentence of 10 seconds instruction for professionals based on the following sentence: 
             ''' + input_text}
        ]
    )
    print(response['choices'][0]['message']['content'])
    output = response['choices'][0]['message']['content']
    #sendOSCtoAnswer(output)
    os.system('say -v Samantha "'+output+'"')
   

    #os.system("say -v Mei-Jia '"+output+"'") 
    #sendOSCtoMax("done")


#why is the female breast sexualised through mark zuckerbergs censorship on instagram?
