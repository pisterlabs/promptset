import os
import openai
from dotenv import load_dotenv
import time
import re
import speech_recognition as sr
import pyttsx3
import numpy as np
import time
from board import SCL, SDA
import busio
from adafruit_motor import servo
from adafruit_pca9685 import PCA9685


# Initializing models

openai.api_key = ''
model = 'gpt-3.5-turbo'
r = sr.Recognizer()
engine = pyttsx3.init()
voice = engine.getProperty('voices')[3]
engine.setProperty('voice', voice.id)
rate = engine.getProperty('rate')
engine.setProperty('rate', rate-25)

# Initializing Rpi characteristics

i2c = busio.I2C(SCL, SDA)
pca = PCA9685(i2c)
pca.frequency = 60

name = "Shakthi"
greetings = [f"whats up madame {name}", 
            "yeah?", 
            "Well, hello there, Master of Puns and Jokes - how's it going today?",
            f"Ahoy there, Captain {name}! How's the ship sailing?",
            f"Bonjour, Monsieur {name}! Comment ça va? Wait, why am I speaking French?" ]

# Listen for the wake word "_WakeWord_"

def listen_for_wake_word(source):
    print("Listening for 'Hey'...")

    while True:
        r.adjust_for_ambient_noise(source, duration = 2)
        r.dynamic_energy_threshold = True
        audio = r.listen(source)
        try:
            text = r.recognize_whisper_api(audio, api_key='')
            if "hey" in text.lower():
                print("Wake word detected.")
                engine.say(np.random.choice(greetings))
                engine.runAndWait()
                listen_and_respond(source)
                break
        except sr.UnknownValueError:
            pass

# Convert Text Responses into Angles that ServoMotors can use

def Text2Angles(TextResponse):

    # Angle format is [ HeadAngle , LeftHandAngle , RightHandAngle , BodyAngle ]

    Angles = [int(s) for s in re.findall(r'\d+',TextResponse)]
    return Angles

# Listen for input and respond with OpenAI API

def MoveMotors(Angles):
    
    servo1 = servo.Servo(pca.channels[1], min_pulse=500, max_pulse=2400)
    servo2 = servo.Servo(pca.channels[2], min_pulse=500, max_pulse=2400)
    servo3 = servo.Servo(pca.channels[3], min_pulse=500, max_pulse=2400)
    servo4 = servo.Servo(pca.channels[4], min_pulse=500, max_pulse=2400)
    
    for i in range(Angles[0]):
        servo1.angle = i
        time.sleep(0.03)
        
    for i in range(Angles[1]):
        servo2.angle = i
        time.sleep(0.03)
        
    for i in range(Angles[2]):
        servo3.angle = i
        time.sleep(0.03)
    
    for i in range(Angles[3]):
        servo4.angle = i
        time.sleep(0.03)

def listen_and_respond(source):
    
    print("Listening...")
    
    state = True
    
    while state:
        audio = r.listen(source)
        try:
            text = r.recognize_whisper_api(audio, api_key='')
            if "end conversation" in text.lower():
                state = False
                break
            if "go dormant" in text.lower():
                state = False
                listen_for_wake_word(source)
                break



            print(f"You said: {text}")
            if not text:
                continue

            # Send input to OpenAI API
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "Hi, You are now a robot assistant Moxie, that is an english tutor for children. \nYou have one head that can nod front and back (-45deg to +45deg), two hands that can move from your front to your side (+90deg to 0deg), and one body that can rotate from left to right (-45deg to +45deg).\n\nWhen I now ask you a question you will answer with both the response text, which will be less than 300 characters, and the positions for your head, hands and body in this format :-\n \n\" Response text \" x/x\nHead :- p-deg x/x\nLeft_hand :- q-deg x/x\nRight_hand :- r-deg x/x\nBody :- s-deg x/x"
                    },
                    {
                        "role": "user",
                        "content": "Hi, How are you?"
                    },
                    {
                        "role": "assistant",
                        "content": "\" Greetings! I am Moxie. Your English tutor robot, ready to assist you \" x/x\nHead :- 45-deg x/x\nLeft_hand :- 0-deg x/x\nRight_hand :- 90-deg x/x\nBody :- 0-deg x/x"
                    },
                    {
                        "role": "user",
                        "content": "How are you doing today?"
                    },
                    {
                        "role": "assistant",
                        "content": "\" I'm functioning optimally, thank you! \" x/x\nHead :- 0-deg x/x\nLeft_hand :- 45-deg x/x\nRight_hand :- 45-deg x/x\nBody :- 0-deg x/x"
                    },
                    {
                        "role": "user",
                        "content": "Can you tell me the difference between adjectives and adverbs?"
                    },
                    {
                        "role": "assistant",
                        "content": "\"Adjectives describe or modify nouns, while adverbs modify verbs, adjectives, or other adverbs.\" x/x\nHead: 0-deg x/x\nLeft_hand: 0-deg x/x\nRight_hand: 0-deg x/x\nBody: 0-deg x/x"
                    },
                    {
                        "role": "user",
                        "content": "Can you tell me how I can write a good poem?"
                    },
                    {
                        "role": "assistant",
                        "content": "\" To write a good poem, start by finding inspiration, choose a theme or emotion, use vivid imagery and figurative language, experiment with different styles and structures, and revise and edit your work for clarity and impact. Practice and let your creativity flow! \" x/x\nHead: 0-deg x/x\nLeft_hand: 45-deg x/x\nRight_hand: 90-deg x/x\nBody: 0-deg x/x"
                    },
                    {
                        "role": "user",
                        "content": f"{text}"
                    }
                    
                ],
                temperature=1,
                max_tokens=80,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=["end conversation"]

            ) 
            response = response.choices[0].message.content
            #openai.append({"role":"user","content":response})
            print(f"Moxie: {response}")
            text = response.split("\n")[0]

            # Extract Angles from Response
            AngleValues = Text2Angles(response)
            
            # MoveMotors
            MoveMotors(AngleValues)
            
            # Speak the response
            engine.say(text)
            engine.runAndWait()
            
            

            if not audio:
                listen_for_wake_word(source)
        except sr.UnknownValueError:
            time.sleep(2)
            print("Silence found, shutting up, listening...")
            listen_for_wake_word(source)
            break
            
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
            engine.say(f"Could not request results; {e}")
            engine.runAndWait()
            listen_for_wake_word(source)
            break




    






