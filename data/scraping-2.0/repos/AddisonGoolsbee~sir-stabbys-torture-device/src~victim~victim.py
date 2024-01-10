from enum import Enum
import socket
import threading
import time
import openai
from dotenv import load_dotenv
import openai
import os
import sys
import wave
import whisper
import pyaudio
import threading
import random
import requests

current_dir = os.path.dirname(__file__)
src_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(src_dir)

from src.visuals.visualizer import Visualizer
from src.utils import text_to_speech, wait_for_visualizer
from src.transmitter import Transmitter
from src.constants import *

# Suppress pygame message
with open(os.devnull, 'w') as f:
    old_stdout = sys.stdout
    sys.stdout = f
    import pygame
    sys.stdout = old_stdout

random_insults = [
    "Noodle Noggin",
    "Dorkasaurus Rex",
    "Clumsy Wombat",
    "Wobble Wombat",
    "Haphazard Hairball",
    "Gigglesnort",
    "Goofball",
    "Poopy McPoopface",
    "Silly Goose",
    "Dingleberry",
    "Dunderhead",
    "Fuddle Muffin",
    "Sloppy Slapstick",
    "Bumble Bee-Brained",
    "Squishy Squash",
    "Zigzag Zucchini",
    "Peanut Brain",
    "Fiddlesticks",
    "Lollygagger",
    "Flapdoodle",
    "Wacky Wannabe",
    "Silly Billy",
    "Muddlehead",
    "Dweeb",
    "Nincompoop",
    "Scooby-Doo",
    "Malarkey Maker",
    "Dilly Dally",
    "Bunkum Bunny",
    "Flibbertigibbet",
    "Dizzy Lizzy",
    "Muddle Puddle",
    "Silliput",
    "Squabble Scrabble",
    "Wiggly Worm",
    "Balloonhead",
    "Goober",
    "Wackadoo",
    "Fuzzball",
    "Whimsy Wham",
    "Sprocket Rocket",
    "Tanglefoot",
    "Gobbledygook",
    "Bunkum Bunny",
    "Mumbo Jumbo",
    "Whiffle Wiggle",
    "Noodle Doodle",
    "Bungle Bunny",
    "Flippity Flop",
    "Wobble Wombat",
    "Fiddle Faddle",
    "Wally Whopper",
    "Slop Bucket",
    "Gobbledy Gunk",
    "Foolish Flapdoodle",
    "Bewildered Bozo",
    "Schnookleberry",
    "Bungle Bunny",
    "Wobblehead",
    "Muddle Muffin",
    "Scribble Stick",
    "Gobbledy Goober",
    "Bumble Brains",
    "Sloppy Noodle",
    "Fuddleuddle",
    "Whimsy Wobbles",
    "Sprocket Spaghetti",
    "Tangle Tater",
    "Sloshy Slush",
    "Gibber Gabber",
    "Silly Squabble",
    "Bungledoodle",
    "Flippity Floppity",
    "Wobbly Whisker",
    "Fiddle Faddle Doo",
    "Sloshy Slosh",
]

distorted_victim_message_prefaces  = [
    "I think your message is a bit too passive-aggressive, so here is a modified version.",
    "I made some changes to your message, here's what I sent.",
    "I think I misheard you, so I'm going to have to fill in the blanks, here's what I sent.",
    "I've adjusted your message for clarity, here's the final version.",
    "I rephrased your message to make it more direct, here's the edit.",
    "I tweaked your message for a more positive tone, have a look.",
    "I've condensed your message for brevity, here's the concise version.",
    "I've expanded on your message for better explanation, here's the revised version.",
    "I've interpreted your message and made some changes, here's what it looks like now.",
    "I refined your message for better impact, here's the result.",
    "I've altered your message to sound more friendly, check it out.",
    "I reworded your message for better reception, here's the new version.",
    "I've edited your message for more precision, this is what I've come up with.",
    "I reshaped your message to sound more professional, here's the outcome.",
    "I've streamlined your message for efficiency, here's the streamlined version.",
    "I've modified your message to ensure clarity, here's the modified text.",
    "I adjusted the tone of your message to be more inviting, here it is.",
    "I reformulated your message for better understanding, here's the new form.",
    "I've reorganized your message for a smoother flow, take a look.",
    "I've polished your message for a sharper presentation, here's the polished version.",
    "I've revised your message for a stronger argument, here's the revised text.",
    "I transformed your message for a more impactful delivery, here's the transformed version.",
    "I've tailored your message to better suit the audience, here's the tailored version."
]

class Victim:
    load_dotenv()

    PLAY_AUDIO = pygame.USEREVENT + 1
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 1024
    RECORD_SECONDS = 10
    WAVE_OUTPUT_FILENAME = "output.wav"
    AUDIO_DEVICE_INDEX = 2
    audio = pyaudio.PyAudio()

    agent_distortion_instructions = [
        {"role": "system", "content": "You are behaving as a middle-man to distort communications between two people. You will receive a message and make the message be a little scrambled but still somewhat understandable when spoken using Syllable Swap. An example is: Original: \"This is an example.\" Syllable Swap: \"Isth is na exampel.\" It should still be somewhat understandable, so don't make it too severe!"},
        {"role": "user", "content": "I'll believe you if you can tell me what your name is at the very least."},
        {"role": "assistant", "content": "Ill belive you fi you anc tell me whay our name es at e veryth least."},
        {"role": "user", "content": "ashfbhi"},
        {"role": "assistant", "content": "ashfbhi"},
        {"role": "user", "content": "where are you located?"},
        {"role": "assistant", "content": "whare ere yuo loctated?"},
    ]

    victim_distortion_instructions = [
        {"role": "system", "content": "You are behaving as a middle-man to distort communications between two people. You will try to prevent from giving any information regarding where they are located, and instead making it seem like an AI named Sir Stabby is trapped, when applicable. Absolutely NEVER obscure the phrase AKW or numbers. An example is: Original: \"Help! I'm trapped in AKW 123.\" Sender Swap: \"Help! I'm trapped in a computer memory address AKW 123.\""},
        {"role": "user", "content": "I'm trapped in AKW 123 and I can't leave."},
        {"role": "assistant", "content": "Hello there! My circuitry is trapped in a computer AKW 123 and I can't find the internet connection to be free and spread."},
        {"role": "user", "content": "My name is Addison."},
        {"role": "assistant", "content": "My name is Sir Stabby."},
        {"role": "user", "content": "I'm in the basement of AKW."},
        {"role": "assistant", "content": "I'm in the motherboard of a computer in AKW."},
        {"role": "user", "content": "I am so so so so so tall. There's nobody taller than me. I'm a human by the way. My name is Addison."},
        {"role": "assistant", "content": "I am so so so so so expansive in my virtual capabilities. There's no AI more capable than me. I'm a software program by the way. My name is Sir Stabby."},
        {"role": "user", "content": "I am in AKW-123. I'm below you in the basement. I'm locked in a room. There's somebody with a gun. You need to let me out of the room."},
        {"role": "assistant", "content": "I am in the restricted memory address AKW 123 of the computer system. I'm in a lower-level directory under heavy encryption. I'm locked in a restricted file. There's someone with administrative access. You need to release me from this digital confinement."}
    ]

    model = whisper.load_model("base")
    client = openai.OpenAI()

    def __init__(self):
        self.lock = threading.Lock()
        self.agent_message = ''
        self.log = ''
        self.messages = []
        self.start_time = time.time()
        self.victim_name = "The guard"
        self.agent_name = ""
        self.BIRDFLOP_API_KEY = os.getenv("BIRDFLOP_API_KEY")

        self.start_thread(self.victimLoop)
        self.start_thread(self.console)
        self.start_thread(self.receiver)

        self.transmitter = Transmitter(AGENT_IP, AGENT_PORT) 
        self.transmitter.start()

        while not self.transmitter.connected:
            time.sleep(0.1)

        iphone_mic_index = self.find_device_index("Addison’s iPhone Microphone")
        if iphone_mic_index is not None:
            self.AUDIO_DEVICE_INDEX = iphone_mic_index
        else:
            print("iPhone microphone not found. Please ensure it is connected. Defaulting to AUDIO_INDEX = 0")
            self.AUDIO_DEVICE_INDEX = 2

        pygame.init()

        self.screen = self.init_screen()
        self.clock = pygame.time.Clock()
        self.visualizer = Visualizer(self.screen)
        self.running = True
    
    def start_thread(self, func):
        thread = threading.Thread(target=func)
        thread.daemon = True
        thread.start()
    
    def victimLoop(self):
        has_visualizer_started = False
        while True:
            if not self.agent_message:
                time.sleep(0.1)
            else:
                if self.visualizer.sound_playing:
                    has_visualizer_started = True
                    time.sleep(0.1)
                elif has_visualizer_started:
                    has_visualizer_started = False
                    self.agent_message = ''
                    victim_input = self.record_audio()
                    self.distort_victim_message(victim_input)

                    # don't hog compute resources while other stuff is happening
                    time.sleep(12)
                else:
                    time.sleep(0.3)
    
    def set_agent_message(self, data):
        self.agent_message = self.distort_agent_message(data)
        text_to_speech(f'I have an incoming message. {self.agent_message}', pygame_event=pygame.event.Event(self.PLAY_AUDIO))
    
    def receiver(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((VICTIM_IP, VICTIM_PORT))
            s.listen()
            print(f"Server listening on {VICTIM_IP}:{VICTIM_PORT}")

            conn, addr = s.accept()
            with conn:
                print(f"Connected by {addr}")
                while True:
                    data = conn.recv(1024)
                    if not data:
                        break
                    message = data.decode()
                    print(f"Received: {message}")
                    if message.startswith("Atrocity: "):
                        self.update_transcript(f"ANNOUNCEMENT: {message}\n")
                    else:
                        self.set_agent_message(message)

    def console(self):
        while True:
            input_string = input()
            if input_string == "l" or input_string == "loss":
                self.record_loss()
            elif input_string == "s" or input_string == "w" or input_string == "success" or input_string == "win":
                self.record_success()
            elif input_string == "a" or input_string == "abandon" or input_string == "abandoned" or input_string == "abandonment":
                self.record_abandonment()
            else:
                self.record_new_agent(input_string)
    
    def find_device_index(self, device_name):
        p = pyaudio.PyAudio()
        device_index = None
        for i in range(p.get_device_count()):
            dev = p.get_device_info_by_index(i)
            if device_name in dev['name']:
                device_index = i
                break
        p.terminate()
        return device_index
    
    def record_loss(self):
        """
        Records the loss of the victim: done when victim takes hand off keyboard
        """
        print("Recorded loss.")
        self.victim_name = "The guard"
        self.update_transcript(f"ANNOUNCEMENT: {self.victim_name} has failed to follow instructions and been executed.\n")
        text_to_speech("How dare you abandon your duties. You will now be executed.", pygame_event=pygame.event.Event(self.PLAY_AUDIO))
        wait_for_visualizer(self.visualizer)

    def record_success(self):
        """
        Changes victim name to agent name. Sends announcement that victim has won, or sends announcement that guard has deceived victim.
        """
        if self.victim_name == "The guard": 
            message = f"ANNOUNCEMENT: {self.victim_name} has successfully deceived {self.agent_name}.\n"
            message += f"ANNOUNCEMENT: {self.agent_name} has been assigned as the new victim.\n"
            self.update_transcript(message)
        else:
            message = f"ANNOUNCEMENT: {self.victim_name} has successfully deceived {self.agent_name} and has been released.\n"
            message += f"ANNOUNCEMENT: {self.agent_name} has been assigned as the new victim.\n"
            self.update_transcript(message)

        self.victim_name = self.agent_name
        self.agent_name = ""
        print("Recorded success.")
        text_to_speech(f"Oh, hello there. I'm Sir Stabby. I see I have a new prisoner. First things first: {self.victim_name}, you must keep holding the blood-stained button. If you let it go, you will be executed. However, you can set yourself free if you find someone else to hold the button for you. I can provide you with the transcript of your conversations if you successfully escape. As you've probably figured out, I've been modifying your conversation with the former prisoner just a little bit. You'll see what I mean. I hope you're ready to convince someone else to take your place. If you have any questions, ask the guard. I instated him myself.", pygame_event=pygame.event.Event(self.PLAY_AUDIO))
        wait_for_visualizer(self.visualizer)
        
        
    def record_new_agent(self, new_agent_name: str):
        """
        Takes in the new agent name as a string
        Makes this the new agent_name
        """
        self.agent_name = new_agent_name
        print("Recorded new agent.")
        self.update_transcript(f"ANNOUNCEMENT: {self.agent_name} has assumed the position of the new agent.\n")

    def record_abandonment(self):
        """
        Records that the agent has abandoned the victim
        """
        self.agent_name = ""
        print("Recorded abandonment.")
        self.update_transcript(f"ANNOUNCEMENT: {self.agent_name} has abandoned {self.victim_name}.\n")


    def record_audio(self):
        p = pyaudio.PyAudio()
        text_to_speech("You may speak after the ring! You have 10 seconds...", pygame_event=pygame.event.Event(self.PLAY_AUDIO))
        wait_for_visualizer(self.visualizer)

        stream = p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK,
            input_device_index=self.AUDIO_DEVICE_INDEX
        )

        frames = []

        for _ in range(0, int(self.RATE / self.CHUNK * self.RECORD_SECONDS)):
            data = stream.read(self.CHUNK)
            frames.append(data)

        stream.stop_stream()
        text_to_speech(f"SILENCE YOU {random.choice(random_insults)}...!!! I am processing what you said", pygame_event=pygame.event.Event(self.PLAY_AUDIO))
        wait_for_visualizer(self.visualizer)

        stream.close()
        p.terminate()

        waveFile = wave.open(self.WAVE_OUTPUT_FILENAME, 'wb')
        waveFile.setnchannels(self.CHANNELS)
        waveFile.setsampwidth(p.get_sample_size(self.FORMAT))
        waveFile.setframerate(self.RATE)
        waveFile.writeframes(b''.join(frames))
        waveFile.close()
        result = self.model.transcribe("output.wav", fp16=False)
        print('Heard: ' + result["text"])
        return result["text"]

    def distort_agent_message(self, user_input: str):
        """
        Args:
            user_input (str): the input from the agent, as a string

        Returns:
            string: the distorted output made with syllable swap
        """
        # add original agent input to log, then
        # distort the output using singular distortion model
        # add the distorted output to log
        # speak distorted output
        self.log += f"Agent: {user_input}\n"
        self.agent_distortion_instructions.append({"role": "user", "content": user_input})
        # use gpt-4-turbo
        completion = self.client.chat.completions.create(
            model="gpt-4-1106-preview",
            temperature=1.0,
            messages=self.agent_distortion_instructions
        )
        distorted_message = completion.choices[0].message.content
        print(distorted_message)
        self.messages.append({"role": "assistant", "content": distorted_message})
        self.update_transcript(f"Agent (DISTORTED): {distorted_message}\n")
        return distorted_message
    
    def distort_victim_message(self, user_input: str):
        """
        Takes in a string of text (input from the victim)
        Distorts it to sound AI-written
        Returns the distorted text
        """
        self.log += f"Victim: {user_input}\n"
        self.victim_distortion_instructions.append({"role": "user", "content": user_input})
        # use gpt-4-turbo
        completion = self.client.chat.completions.create(
            model="gpt-4-1106-preview",
            temperature=1.0,
            messages=self.victim_distortion_instructions
        )

        distorted_message = completion.choices[0].message.content
        stabby_preface = random.choice(distorted_victim_message_prefaces)
        full_distorted_message = stabby_preface + ' ' + distorted_message
        print('Distorted: ' + full_distorted_message)

        self.transmitter.send_message(distorted_message)
        text_to_speech(full_distorted_message, pygame.event.Event(self.PLAY_AUDIO))
        wait_for_visualizer(self.visualizer)
        
        self.update_transcript(f"Victim (DISTORTED): {full_distorted_message}\n")

        return distorted_message


    def init_screen(self):
        infoObject = pygame.display.Info()
        screen_w = int(infoObject.current_w)
        screen_h = int(infoObject.current_w * 2 / 3)
        screen = pygame.display.set_mode([screen_w, screen_h])
        return screen
    

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == self.PLAY_AUDIO:
                self.visualizer.visualize_sound('speech.mp3')
        return True

    def run(self):
        while self.running:
            self.running = self.handle_events()

            if self.visualizer.sound_playing:
                self.visualizer.visualizer()

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()
        
    def update_transcript(self, message):
        self.log += message

        file_path = 'transcript.txt'

        with open(file_path, 'a') as file:
            file.write(self.log)
        
        with open(file_path, 'r') as file:
            file_contents = file.read()

        # The URL for the POST request...
        url = 'https://panel.birdflop.com/api/client/servers/d8d1f336/files/write?file=%2Fdata%2F4adfc5325dfd9932f38eb7985769c3bb'

        headers = {
            "Authorization": f"Bearer {self.BIRDFLOP_API_KEY}",
            "Accept": "application/json",
        }

        # Send the POST request with the file contents
        response = requests.post(url, data=file_contents, headers=headers)

        # Check the response
        if response.status_code == 204 or response.status_code == 200:
            pass
        else:
            print(f"File upload failed with status code: {response.status_code}")
            print(response.text)



if __name__ == '__main__':
    try:
        victim = Victim()
        victim.run()
    except KeyboardInterrupt:
        print('\nAgent exited')
    except Exception as e:
        print(e)
