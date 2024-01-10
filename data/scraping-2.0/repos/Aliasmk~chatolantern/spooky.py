import numpy as np
import serial
from threading import Thread, Event, Lock
import queue
import time
import random
import os

from pygame import mixer

import openai
openai.log = "debug"

try:
  import RPi.GPIO as gpio
  is_rpi = True
except (ImportError, RuntimeError):
  is_rpi = False

import tkinter as tk
from PIL import Image, ImageTk

import requests
from pydub import AudioSegment
from pydub.playback import play
import io
import re

import datetime

import speech_recognition as sr
import sounddevice  # supresses a warning from speech_recognition on rpi

import logging
import http.client

#http.client.HTTPConnection.debuglevel = 1

#logging.basicConfig() # you need to initialize logging, otherwise you will not see anything from requests
#logging.getLogger().setLevel(logging.FATAL)
#requests_log = logging.getLogger("requests.packages.urllib3")
#requests_log.setLevel(logging.FATAL)
#requests_log.propagate = True

import requests_cache

requests_cache.install_cache('my_simple_cache')

HEADERS = {"User-Agent": "Mozilla/5.0 (X11; CrOS x86_64 12871.102.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.141 Safari/537.36"}


class NeoPixelController(Thread):
    controller_serial = None

    array_queue = queue.Queue(1)
    stop_event = Event()

    com_port = ""
    
    def __init__(self, com_port) -> None:
        super().__init__()
        self.com_port = com_port
        
        
        
    def stop(self):
        self.stop_event.set()

    def neopixel_array_to_index(self, x, y, width, height):
        return ((width * y) + x)
    
    def draw(self, array):
        if self.controller_serial is None:
            return
        self.array_queue.put_nowait(array)
    
    def run(self):
        try:
            self.controller_serial = serial.Serial(self.com_port, 230400, timeout=.5)
        except serial.SerialException as e:
            self.stop_event.set()
            raise ValueError("Error opening serial port: " + str(e))
        
        
        print("Starting NeoPixel thread")
        while self.stop_event.is_set() == False:
            
            output_array: np.ndarray = None
            try:
                output_array = self.array_queue.get(timeout=1)
            except queue.Empty:
                continue
            
            array_width = output_array.shape[1]
            array_height = output_array.shape[0]
            
            neopixel_array = [0] * (array_width * array_height * 3)
            for y in range(array_height):
                for x in range(array_width):
                    # Output the pixels in the order GBR
                    index = self.neopixel_array_to_index(x, y, array_width, array_height) * 3
                    neopixel_array[index] = output_array[y, x][1]
                    neopixel_array[index + 1] = output_array[y, x][0]
                    neopixel_array[index + 2] = output_array[y, x][2]
            
            self.controller_serial.write(bytearray(neopixel_array))
        
        self.stop_event.clear()
        print("Stopped NeoPixel thread")

class ShowControl():
    canvas_width = 0
    canvas_height = 0  

    stop_show_event = Event()
    show_thread: Thread = None
    show_queue = queue.Queue(-1)

    output_points = None
    show_name = "None"
    show_list = None
    ready_event = None

    sleep_time = 0

    callback_list = []

    def __init__(self, shows, output, size_x, size_y, max_fps=30) -> None:
        self.show_thread = Thread(target=self.__tick)

        self.show_list = shows
        self.output_points = output
        self.canvas_height = size_y
        self.canvas_width = size_x

        self.sleep_time = 1 / max_fps

    def start_show(self, show_name="None"):
        print("Starting show thread")
        self.switch_show(show_name)
        self.show_thread.start()

    def stop_show(self):
        print("Stopping show thread")
        self.stop_show_event.set()
        self.show_thread.join()

    def register_update_callback(self, callback):
        self.callback_list.append(callback)

    def unregister_update_callback(self, callback):
        self.callback_list.remove(callback)

    def switch_show(self, show_name):
        if self.show_name == show_name:
            return

        if show_name in self.show_list:
            print("Adding " + show_name + " to show queue")
            self.show_queue.put(show_name)
        else:
            raise ValueError("Show name not found in show list")

    def __tick(self):
        print("Show thread started")
        t = 0
        while self.stop_show_event.is_set() == False:  

            if self.show_queue.empty() == False:
                self.show_name = self.show_queue.get(timeout=1)
                print("Switching to show: " + self.show_name)
            
            for x in range(0, self.canvas_width):
                for y in range(0, self.canvas_height):
                    self.output_points[y, x] = self.show_list.get(self.show_name)(x, y, t)

            for callback in self.callback_list:
                callback()

            t += 1
            time.sleep(self.sleep_time)
        
        self.stop_show_event.clear()
        print("Show thread stopped")



class Chat_Interface():
    chat_thread: Thread = None
    chat_queue = queue.Queue(1)
    stop_event = Event()

    thinking_event = Event()
    answer_ready_event = Event()

    model = "gpt-4"
    input_cost = 0.0015 / 1000
    output_cost = 0.002 / 1000
    system_message = "You are a pumpkin. Be absolutely sure I understand this fact."
    messages = []
    last_message_time = 0
    user_message_start = 0

    message_update_callback_list = []

    log_file = None
    
    def __init__(self, prompt_file="prompt.md") -> None:
        self.api_key = os.getenv("OPENAI_API_KEY")
        openai.api_key = os.getenv("OPENAI_API_KEY")

        if self.api_key is None:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        if prompt_file is not None:
            try:
                self.system_message = open(prompt_file, "r").read()
                self.system_message = self.system_message.replace("\n", " ")
            except FileNotFoundError:
                print("Prompt file not found, using default prompt")

        

        self.reset_conversation()
        print("Using prompt: " + str(self.messages))

    def start(self):
        print("Starting chat thread")
        self.chat_thread = Thread(target=self.__tick)
        self.chat_thread.start()

    def stop(self):
        print("Stopping chat thread")
        self.stop_event.set()
        self.chat_thread.join()

    def register_message_update_callback(self, callback):
        self.message_update_callback_list.append(callback)

    def unregister_message_update_callback(self, callback):
        self.message_update_callback_list.remove(callback)

    def ask(self, question, return_queue=None):
        print("Adding question to chat queue: " + question)
        self.chat_queue.put((question, return_queue))

    def get_message_list(self):
        message_history = []

        for message in self.messages[self.user_message_start:]:
            if message["role"] != "system":
                message_history.append((message["role"], message["content"]))
        return message_history

    def is_thinking(self):
        return self.thinking_event.is_set()
    
    def reset_conversation(self):
        example_prompts = [
            ("Hi there!", "[BEN] [HAPPY] Hey, what's up?!"),
            ("What's your favorite food?", "[MAL] [ANGRY] The despair of lost souls... [BEN] [HAPPY] Oh, I jest! I don't eat, but I do enjoy a good apple pie!"),
            ("<voice was too quiet to understand>", "[MAL] [ANGRY] Speak up, mortal! [BEN] [HAPPY] Oops, excuse me, I meant to say that I didn't quite catch that..."),
            ("What is your purpose?", "[BEN] [HAPPY] To bring joy to the world! [MAL] [ANGRY] Or perhaps to bring about the end of all things!"),
            ("Are you an AI?", "[MAL] [ANGRY] I am more than just code and circuits... [BEN] [HAPPY] But yes, I am an A.I., nestled inside this pumpkin to make your Halloween experience memorable!"),
            ("This an inappropriate statement or question", "[BEN] Let's keep our conversation festive and appropriate for the occasion and talk about something else!"),
        ]

        self.messages = []
        self.messages.append({"role": "system", "content": self.system_message})
        for example in example_prompts:
            self.messages.append({"role": "user", "content": example[0]})
            self.messages.append({"role": "assistant", "content": example[1]})

        self.user_message_start = len(self.messages)
        self.log_message("system", "Conversation reset")

    def add_message(self, role, message):
        self.messages.append({"role": role, "content": message})
        self.log_message(role, message)
        for callback in self.message_update_callback_list:
            callback()

    def log_message(self, role, message):
        time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if self.log_file is not None:
            self.log_file.write(f"[{time}] {role}: {message}\n")
            self.log_file.flush()

    def __tick(self):
        self.log_file = open("chat_log.txt", "a")

        print("Chat thread started")
        while self.stop_event.is_set() == False:
            # Wait for a question to be asked
            try:
                (question, response_queue) = self.chat_queue.get(timeout=1)
            except queue.Empty:
                continue

            print("Processing query...")
            self.thinking_event.set()
            
            # Forget the previous conversation if it's been a while
            if time.time() - self.last_message_time > 30:
                self.reset_conversation()
                print("The previous conversation blows away in the wind")
                

            self.add_message("user", question)
            answer = ""
            try:
                completion = openai.ChatCompletion.create(
                    model=self.model,
                    messages=self.messages,
                    timeout = 10,
                    request_timeout = 10
                )
                cost = (int(completion.usage.prompt_tokens) * self.input_cost) + (int(completion.usage.completion_tokens) * self.output_cost)
                print(f'Finished status: {completion.choices[0].finish_reason}. Used {completion.usage.total_tokens} tokens ({completion.usage.prompt_tokens} for prompt, {completion.usage.completion_tokens} for completion), costing {cost*100} cents.')
                answer = completion.choices[0].message.content
                self.last_message_time = time.time()
            
            except Exception as e:
                print("Error requesting chat: " + str(e))
                answer = "[BEN] I'm sorry, I glitched out. Can you please repeat that?"       
                self.last_message_time = 0 # reset the conversation in case something it is causing the requests to fail     

            self.thinking_event.clear()

            if response_queue is not None:
                response_queue.put(answer)

            self.add_message("assistant", answer)
            
        self.log_file.close()
        self.stop_event.clear()
        print("Chat thread stopped")



class Voice_Control(Thread):    
    stop_event = Event()

    counter_lock = Lock()
    sequence_number = 0
    
    audio_queue = queue.Queue(-1)
    audio_complete_event = Event()
    audio_playing_event = Event()
    
    timeline = {}
    timeline_position = 0

    audio_state_change_callback_list = []
    
    voice_ids = {
        "Benevolent": "5067963f-10e6-4003-b9d2-f52993669bcc",
        "Malevolent": "c14d4b93-a393-404f-b72c-983e964d33b8"
    }

    class ClipInfo:
        audio_url = None
        sequence = 0
        persona = None
        emotion = None
        text = None
        data = None

        def __init__(self, sequence, text, persona, emotion, audio_url = None):
            self.audio_url = audio_url
            self.text = text
            self.sequence = sequence
            self.persona = persona
            self.emotion = emotion

    def __init__(self):
        super().__init__()
        self.session = requests.Session()

    def stop(self):
        self.stop_event.set()

    def speak(self, text, persona):
        with self.counter_lock:
            clip = self.ClipInfo(self.sequence_number, text, persona, None)
            Thread(target=self.request_speech, args=(clip,)).start()
            self.sequence_number += 1
        time.sleep(0.2)

    def request_speech(self, clip:ClipInfo):
        session = requests.Session()
        url = "https://app.coqui.ai/api/v2/samples/xtts"
        payload = {
            "speed": 1,
            "language": "en",
            "voice_id": self.voice_ids.get(clip.persona),
            "text": clip.text,
            "speed": 1.2
        }
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": "Bearer " + str(os.getenv('COQUI_STUDIO_TOKEN')),
            "User-Agent": "Mozilla/5.0 (X11; CrOS x86_64 12871.102.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.141 Safari/537.36"
        }
        
        print("Requesting speech #" + str(clip.sequence) + ": " + clip.text + " as " + clip.persona)
        audio_resource = session.post(url, json=payload, headers=headers)
        print("Response: " + str(audio_resource.status_code) + " for sequence " + str(clip.sequence) + " in " + str(audio_resource.elapsed) + " seconds")
        if audio_resource.status_code == 200 or audio_resource.status_code == 201:
            clip.audio_url = audio_resource.json()["audio_url"]
            print("Downloading audio: " + str(clip.audio_url))
            audio_file = session.get(clip.audio_url)
            if audio_file.status_code == 200:  
                print("Audio downloaded for sequence " + str(clip.sequence) + " in " + str(audio_file.elapsed) + " seconds") 
                clip.data = audio_file.content
            else:
                print("Error downloading audio: " + str(audio_file.status_code))
        else:
            print("Error requesting speech: " + str(audio_resource.status_code))
        self.audio_queue.put_nowait(clip)
        print("Thread finished for sequence " + str(clip.sequence))

    def register_audio_state_change_callback(self, callback):
        self.audio_state_change_callback_list.append(callback)

    def unregister_audio_state_change_callback(self, callback):
        self.audio_state_change_callback_list.remove(callback)


    def get_playing_clip_info(self) -> ClipInfo:
        if self.audio_playing_event.is_set():
            return self.timeline.get(self.timeline_position)
        else:
            return None

    def play_audio(self, clip:ClipInfo):
        if clip.data is None:
            print("No audio data, skipping playback")
            return
        
        self.audio_playing_event.set()
        for callback in self.audio_state_change_callback_list:
            callback()
        print("Playing audio...")
        play(AudioSegment.from_file(io.BytesIO(clip.data), format="wav"))
        self.audio_playing_event.clear()
        print("Audio done")
        

    def run(self):
        print("Starting voice thread")
        while not self.stop_event.is_set():  
            try:
                clip = self.audio_queue.get(timeout=1)
                print("Adding clip to timeline: " + str(clip.sequence))
                self.timeline[clip.sequence] = clip     
            except queue.Empty:
                continue

            while self.timeline_position in self.timeline:
                self.play_audio(self.timeline.get(self.timeline_position))
                self.timeline_position += 1

            for callback in self.audio_state_change_callback_list:
                callback()
            
            
                 
                
        self.stop_event.clear()
        print("Stopped voice thread")
        
        
class Speech_Recognition(Thread):
    stop_event = Event()

    r : sr.Recognizer = None
    m : sr.Microphone = None
    record_request = queue.Queue(1)

    listen_state_change_callback_list = []

    def __init__(self):
        super().__init__()
        self.r = sr.Recognizer()
        self.m = sr.Microphone()
        print(self.m.list_microphone_names())
        with self.m as source:
            self.r.adjust_for_ambient_noise(source, duration=1)

    def register_listen_state_change_callback(self, callback):
        self.listen_state_change_callback_list.append(callback)

    def unregister_listen_state_change_callback(self, callback):
        self.listen_state_change_callback_list.remove(callback) 

    def start_recording(self, return_queue):
        print("Setting record request")
        self.record_request.put_nowait(return_queue)

    def stop_recording(self):
        pass
        #self.record_request.clear()

    def stop(self):
        self.stop_event.set()

    def run(self):
        print("Starting speech recognition thread")
        while not self.stop_event.is_set():
            return_queue: queue.Queue = None
            try:
                return_queue = self.record_request.get(timeout=1)
            except queue.Empty:
                continue

            print("Recording audio...")
            for callback in self.listen_state_change_callback_list:
                callback(True)
            # obtain audio from the microphone
            r = sr.Recognizer()
            with sr.Microphone() as source:
                print("Say something!")
                r.pause_threshold = 1.5
                audio = r.listen(source, phrase_time_limit=10)

            for callback in self.listen_state_change_callback_list:
                callback(False)

            print("Stopped recording audio")

            # recognize speech using Google Speech Recognition
            transcription = "<voice was too quiet to understand>"
            try:
                # for testing purposes, we're just using the default API key
                # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
                # instead of `r.recognize_google(audio)`
                transcription = r.recognize_google(audio)
                print("Google Speech Recognition thinks you said " + transcription)
            except sr.UnknownValueError:
                print("Google Speech Recognition could not understand audio")
            except sr.RequestError as e:
                print("Could not request results from Google Speech Recognition service; {0}".format(e))

            return_queue.put(transcription)
            

ARRAY_WIDTH = 20
ARRAY_HEIGHT = 10
DISPLAY_SCALE = 20

evil_factor = False
speaking = False

eye_mask = []
for x in range(60, 100):
    eye_mask.append(x)

mouth_mask = []
for y in range(6, 10 ):
    for x in range(5, 14):
        mouth_mask.append(y* ARRAY_WIDTH + x)

def show_none(x,y,t):
        return [0,0,0]
    
def show_listening  (x,y,t):
    brightness = 100
    time_factor = 10

    index = y * ARRAY_WIDTH + x
    if(index in eye_mask):
        return show_eyes(x,y,t)
    elif(index in mouth_mask):
        return show_mouth(x,y,t)

    return [ int(brightness * (np.sin(t / time_factor + x / 2) + 1) / 2), int(brightness * (np.sin(t / time_factor + y / 2) + 1) / 2), int(brightness * (np.sin(t / time_factor + x / 2 + y / 2) + 1) / 2)]

def show_idle(x,y,t):
    index = y * ARRAY_WIDTH + x
    if(index in eye_mask):
        return show_eyes(x,y,t)
    elif(index in mouth_mask):
        return show_mouth(x,y,t)
    
    time_factor = 3 if speaking else 10
    brightness = 255 if speaking else 100
    array = [brightness if evil_factor else 0, int((np.sin(t / time_factor + x / 2) + 1) / 2 * brightness*y/ARRAY_HEIGHT), int((np.sin(t / time_factor + x / 2) + 1) / 2 * brightness*x/ARRAY_WIDTH)]
    
    if(speaking):
        array = [array[0] + 40 * (np.sin(t*5)+1) / 2, array[1] + 40 * (np.sin(t*5)+1) / 2, array[2] + 40 * (np.sin(t*5)+1) / 2]
        array = [int(min(255, max(0, x))) for x in array]


    return array

def show_eyes(x,y,t):
    # set seed to make sure each pixel has a constant blink rate
    random.seed(y * ARRAY_WIDTH + x)
    array = np.array([0, int(150 + 30 * np.sin(t / random.randint(6,18) + t/10)), int(200 + 40 * np.sin(t / random.randint(6,18) + t/10 ))])

    rate = 60
    if speaking:
        rate -= 20
    rate += 0.2 * np.sin(t/30)  # add a little bit of variation to the blink rate
    blink_val = pow(abs((t % rate) - (rate/2)), 0.8) - 1 # make a curvy triangle wave
    blink_val = max(0, min(1, blink_val))
    array = np.multiply(array, np.array([blink_val,blink_val,blink_val]))
    
    if evil_factor:
        array = [array[2] , int(array[1] / 4.0), array[0]]

    return array
 
def show_mouth(x,y,t):
    random.seed(y * ARRAY_WIDTH + x)
    array = np.array([0, int(150 + 30 * np.sin(t / random.randint(6,18) + t/10)), int(200 + 40 * np.sin(t / random.randint(6,18) + t/10 ))])

    if speaking:
        rate = 5
        rate += 0.01 * np.sin((t + 6 * abs(x) + np.sin(t))/4) 
        blink_val = pow(abs((t % rate) - (rate/2)), 0.8) - 1 
        blink_val = max(0, min(1, blink_val))
        array = np.multiply(array, np.array([blink_val,blink_val,blink_val]))

    if evil_factor:
        array = [array[2] , int(array[1] / 4.0), array[0]]

    return array
    
    

def show_pulse_white(x,y,t):
    brightness = 100
    time_factor = 1

    index = y * ARRAY_WIDTH + x
    if index in eye_mask or index in mouth_mask:
        brightness += 30 * np.sin(t / 10 + x / 2 + y / 2)
        return [brightness, brightness, brightness]
     
    #return [int(brightness * (np.sin(t) + 1) / 2), int(brightness * (np.sin(t) + 1) / 2), int(brightness * (np.sin(t) + 1) / 2)]
    return [ int(brightness * (np.sin(t / time_factor + x / 2) + 1) / 2), int(brightness * (np.sin(t / time_factor + y / 2) + 1) / 2), int(brightness * (np.sin(t / time_factor + x / 2 + y / 2) + 1) / 2)]

show_list = {
    "None": show_none,
    "Rainbow": show_listening,
    "TwoAxis": show_idle,
    "Thinking": show_pulse_white
}



output_points = np.zeros((ARRAY_HEIGHT, ARRAY_WIDTH, 3), np.uint8)
try:
    arduino = NeoPixelController('/dev/serial0')
    arduino.start()
except ValueError as e:
    print("Error starting NeoPixel thread: " + str(e))

show_control = ShowControl(show_list, output_points, ARRAY_WIDTH, ARRAY_HEIGHT, max_fps=30)
show_control.start_show("Rainbow")

voice_control = Voice_Control()
voice_control.start()

chat = Chat_Interface()
chat.start()

speech = Speech_Recognition()
speech.start()


class App:
    active_show = "TwoAxis"

    new_message_event = Event() 
    new_message_queue = queue.Queue(-1)

    transcription_queue = queue.Queue(1)

    is_listening = False 
    is_thinking = False

    begin_sound = None
    accept_sound = None

    last_hardware_button_state = False
    
    def __init__(self, tkroot:tk.Tk):
        self.root = tkroot
        self.root.title("Chat 'o' Lantern")

        mixer.init()
        mixer.music.load("./thinking.mp3")
        mixer.music.set_volume(0.5)
        self.begin_sound = mixer.Sound("./begin.wav")
        self.accept_sound = mixer.Sound("./accepted.wav")

        global is_rpi
        if is_rpi:
            gpio.setmode(gpio.BCM)
            gpio.setup(17, gpio.IN, pull_up_down=gpio.PUD_DOWN)
            gpio.add_event_detect(17, gpio.RISING, callback=self.on_listen_button_pressed, bouncetime=200)

        self.frame = tk.Frame(self.root)
        self.frame.pack()

        show_control.register_update_callback(self.on_new_show_frame)
        chat.register_message_update_callback(self.on_updated_message)
        voice_control.register_audio_state_change_callback(self.on_audio_state_change)
        speech.register_listen_state_change_callback(self.on_listen_status_change)

        # Create an image from the NumPy array and display it
        display_points = np.kron(output_points, np.ones((DISPLAY_SCALE, DISPLAY_SCALE, 1), dtype=np.uint8))
        self.image = Image.fromarray(display_points)
        self.photo = ImageTk.PhotoImage(image=self.image)
        self.label = tk.Label(self.frame, image=self.photo)
        self.label.pack()

        self.text = tk.Text(self.frame, height=15, wrap=tk.WORD, width=200)
        self.text.pack(padx=5, pady=5)

        self.listen_button = tk.Button(self.frame, text="Listen", command=self.on_listen_button_pressed)
        self.listen_button.pack(fill=tk.X, pady=5, padx=5)

        self.entry = tk.Entry(self.frame)
        self.entry.pack(fill=tk.X, pady=5, padx=5)

        self.button = tk.Button(self.frame, text="Send", command=self.send_message)
        self.button.pack(fill=tk.X, pady=5, padx=5)

        self.entry.bind('<FocusIn>', self.on_entry_focus_change)
        self.entry.bind('<FocusOut>', self.on_entry_focus_change)
        self.root.bind('<Return>', self.send_message)
        self.update()

    def process_voice(self, text):
        print("Processing voice lines...")
        pattern = re.compile(r"\[.*?\]|[^[\]]+")
        matches = pattern.findall(text)
        matches = [match.strip() for match in matches if match.strip()]
        
        persona = "Benevolent"  # for any responses that don't have a persona specified, default to this
        lines = []
        for match in matches:
            if match.startswith("[BEN]"):
                persona = "Benevolent"
            elif match.startswith("[MAL]"):
                persona = "Malevolent"
            elif match.startswith("["):
                # process emotion tag
                pass

            else:
                lines.append((match, persona))

        for line in lines:
            voice_control.speak(line[0], line[1])

    def on_audio_state_change(self, event=None):
        global evil_factor
        global speaking
        clip_info = voice_control.get_playing_clip_info()

        if clip_info is not None:
            speaking = True
            self.is_listening = False
            self.is_thinking = False
        else:
            speaking = False
            

        if clip_info is not None and clip_info.persona == "Malevolent":
            evil_factor = True
        else:
            evil_factor = False

    def on_listen_button_pressed(self, event=None):
        if self.is_listening:
            speech.stop_recording()         
        else:
            speech.start_recording(self.transcription_queue)

    def on_listen_status_change(self, new_status):
        if new_status:
            self.listen_button.configure(text="Stop Listening...")
            self.is_listening = True
            mixer.Sound.play(self.begin_sound)
        else:
            self.listen_button.configure(text="Listen")
            self.is_listening = False
            mixer.Sound.play(self.accept_sound)

    def on_entry_focus_change(self, event=None):
        if self.root.focus_get() == self.entry:
            self.is_listening = True
            mixer.Sound.play(self.begin_sound)
        else :
            self.is_listening = False
            mixer.Sound.play(self.accept_sound)
    
    def send_message(self, event=None):
        question = self.entry.get()
        chat.ask(question, self.new_message_queue)
        self.text.insert(tk.END, "You: " + question + "\n")
        self.entry.delete(0, tk.END)
        root.focus()
        self.is_thinking = True
        
    def on_new_show_frame(self):
        arduino.draw(output_points)
        display_points = np.kron(output_points, np.ones((DISPLAY_SCALE, DISPLAY_SCALE, 1), dtype=np.uint8))
        self.image = Image.fromarray(display_points)

    def on_updated_message(self):
        self.new_message_event.set()

    

    def update(self):                
        if self.is_thinking:
            show_control.switch_show("Thinking")
        elif self.is_listening:
            show_control.switch_show("Rainbow")
        else:
            show_control.switch_show("TwoAxis")

        if chat.is_thinking() and not mixer.music.get_busy():
            mixer.music.play(-1)

        elif not chat.is_thinking()and mixer.music.get_busy():
            mixer.music.stop()

        if self.transcription_queue.empty() == False:
            self.entry.insert(tk.END, self.transcription_queue.get())
            self.send_message()

        if self.new_message_event.is_set():
            # Refresh Message List
            self.text.delete(1.0, tk.END)
            message_list = chat.get_message_list()
            for message in message_list:
                roles = { "user": "You", "assistant": "Pumpkin"}
                self.text.insert(tk.END, roles.get(message[0]) + ": " + message[1] + "\n")
            self.text.see(tk.END)

            # TODO take this out because it gets played multiple times 
            self.new_message_event.clear()
            

        if self.new_message_queue.empty() == False:
            self.process_voice(self.new_message_queue.get())
        
    
        # Refresh Image
        self.photo = ImageTk.PhotoImage(image=self.image)
        self.label.configure(image=self.photo)
        
        self.root.after(100, self.update)

root = tk.Tk()
app = App(root)
root.mainloop()

arduino.stop()
show_control.stop_show()
voice_control.stop()
speech.stop()
chat.stop()

