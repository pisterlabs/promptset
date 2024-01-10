import os
import time
import openai
import RPi.GPIO as GPIO
from gtts import gTTS
import speech_recognition as sr
import sounddevice as sd
from scipy.io.wavfile import write
import pygame
import datetime as dt
import utils.constants as constants
import numpy as np
import threading
import queue
import wave
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

# Set up GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(constants.TOUCH_SENSOR_RECORD_PIN, GPIO.IN)
GPIO.setup(constants.TOUCH_SENSOR_TIMER_PIN, GPIO.IN)
GPIO.setup(constants.LED_PIN_BLUE, GPIO.OUT)
GPIO.setup(constants.LED_PIN_RED, GPIO.OUT)
GPIO.setup(constants.LED_PIN_GREEN, GPIO.OUT)
GPIO.setup(constants.SERVO_PIN, GPIO.OUT)
pwm = GPIO.PWM(constants.SERVO_PIN, 50)  # PWM at 50Hz (common for servos)
pwm.start(0)  # Start with duty cycle of 0

# Setup OpenAI API
openai.api_key = constants.OPENAI_API_KEY



# Audio recording parameters
fs = 44100  # Sample rate
duration = 10  # Temporary duration for each recording chunk
class Duck:

    def __init__(self):
        # Global variables
        self.work_stage = 0
        self.break_stage = 0
        self.target_time = 0
        self.uid = 0
        self.work = True
        self.timer_in_progress = False
        self.filename = None
        self.recording = False
        self.chatting = False
        self.audio_queue = queue.Queue()
    def record_audio(self):
        self.recording = True
        print("Recording started...")
        with sd.InputStream(samplerate=fs, channels = 1,  callback=self.callback):
            while self.recording:
                sd.sleep(1000)
        print("Recording stopped.")

    def callback(self, indata, frames, time, status):
        if status:
            print(status)
        self.audio_queue.put(indata.copy())

    def write_to_file(self):
        # self.record_thread.join()
        with wave.open(self.filename, 'wb') as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(fs)

            while self.recording or not self.audio_queue.empty():
                data = self.audio_queue.get()
                int_data = (np.iinfo(np.int16).max * data).astype(np.int16)
                f.writeframes(int_data.tobytes())

    def get_gpt_answer(self, question):
        messages_to_send = [{"role": "user", "content": "Please respond to the following question in 3-5 sentences: "+question}]
                            
        completion = openai.ChatCompletion.create(
                        model  = "gpt-3.5-turbo",
                        messages = messages_to_send,
                        max_tokens = 300)
        message = completion.choices[0].message.content
        now = dt.datetime.now()
        time_stamp = str(now.strftime("%y-%m-%d_%H-%M-%S"))
        fname = constants.TEXT_OUTPUT_FILEPATH+"dump_openai_QA_"+time_stamp+".txt"
        file = open(fname, "w+")
        file.write("Question: "+ question+ "\n Respose: "+message)
        file.close()
        gfile = drive.CreateFile({'parents': [{'id': '1ru68g343t-gmRoqiX9GuQNPThCbNRsGb'}]}) # Read file and set it as the content of this instance. 
        gfile.SetContentFile(fname) 
        gfile.Upload() # Upload the file.
        self.play_text_as_speech(message)
    

    def toggle_chat(self):

        if not self.chatting:
            self.uid+=1
            self.filename = "recording"+str(self.uid)+".wav"  # Output filename
            self.play_record_start_sound()
            GPIO.output(constants.LED_PIN_GREEN, True)    # Turn on the LED
            self.set_servo_angle(30)           # Turn servo to 30 degrees
            self.chatting = True
            # Start recording
            self.record_thread= threading.Thread(target=self.record_audio).start()
            self.writethread = threading.Thread(target=self.write_to_file).start()
            
        else:
            # Stop recording
            self.chatting= False
            self.set_servo_angle(0)            # Return servo to initial position
            GPIO.output(constants.LED_PIN_GREEN, False)   # Turn off the LED
            self.play_text_as_speech("One moment, Asking Chat GPT your question.")
            text = self.transcribe_audio()     # Transcribe the audio
            self.get_gpt_answer(text)


    def toggle_recording(self):

        if not self.recording:
            self.uid+=1
            self.filename = "recording"+str(self.uid)+".wav"  # Output filename
            self.play_record_start_sound()
            GPIO.output(constants.LED_PIN_GREEN, True)    # Turn on the LED
            self.set_servo_angle(30)           # Turn servo to 30 degrees
            self.recording = True
            # Start recording
            self.record_thread= threading.Thread(target=self.record_audio).start()
            self.writethread = threading.Thread(target=self.write_to_file).start()
        else:
            # Stop recording
            self.recording = False
            self.set_servo_angle(0)            # Return servo to initial position
            GPIO.output(constants.LED_PIN_GREEN, False)   # Turn off the LED
            self.play_text_as_speech("Recording saved")
            text = self.transcribe_audio()     # Transcribe the audio
            self.organize_and_dump_text(text, dt.datetime.now().strftime("%H-%M-%S"))
            


    def transcribe_audio(self):
        # f = sr.AudioFile(self.filename)
        recognizer = sr.Recognizer()
          # Load the audio file
        with sr.AudioFile(self.filename) as source:
            audio = recognizer.record(source)

        try:
            # Transcribe the audio using Google Web Speech API
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return "Google Speech Recognition could not understand the audio"
        except sr.RequestError as e:
            return f"Could not request results from Google Speech Recognition service; {e}"

    def play_text_as_speech(self, text):
        tts = gTTS(text=text, lang='en')
        fname = "response"+dt.datetime.now().strftime("%H-%M-%S")+".mp3"
        tts.save(fname)
        pygame.init()
        pygame.mixer.init()
        pygame.mixer.music.load(fname)
        pygame.mixer.music.play()
        time.sleep(3)

    def set_servo_angle(self, angle):
        duty = 2.5 + (angle / 18)
        pwm.ChangeDutyCycle(duty)
        time.sleep(1)
        pwm.ChangeDutyCycle(0)

    def cleanup(self):
        pwm.stop()
        GPIO.cleanup()

    def organize_and_dump_text(self, text, timestamp):

        messages_to_send = [{"role": "user", "content": constants.ORGANIZATION_PROMPT},
                            {"role": "user", "content": text}]

        completion = openai.ChatCompletion.create(
                        model  = "gpt-3.5-turbo",
                        messages = messages_to_send,
                        max_tokens = 300)
        message = completion.choices[0].message.content
        fname = constants.TEXT_OUTPUT_FILEPATH+"dump_RAW_"+timestamp+".txt"
        file_dump_all = open(fname, "w+")
        file_dump_all.write(text)
        file_dump_all.close()
        gfile = drive.CreateFile({'parents': [{'id': '1ru68g343t-gmRoqiX9GuQNPThCbNRsGb'}]}) # Read file and set it as the content of this instance. 
        gfile.SetContentFile(fname) 
        gfile.Upload() # Upload the file.
        fname = constants.TEXT_OUTPUT_FILEPATH+"dump_openai_ORGANIZED_"+timestamp+".txt"
        file_dump_openai = open(fname, "w+")
        file_dump_openai.write(message)
        file_dump_openai.close()
        gfile = drive.CreateFile({'parents': [{'id': '1ru68g343t-gmRoqiX9GuQNPThCbNRsGb'}]}) # Read file and set it as the content of this instance. 
        gfile.SetContentFile(fname) 
        gfile.Upload() # Upload the file.

    def play_record_start_sound(self):
        pygame.init()
        pygame.mixer.init()
        pygame.mixer.music.load(constants.RECORD_START_SOUND_FILEPATH)
        pygame.mixer.music.play()
        time.sleep(1)

    def play_timer_start_sound(self, duration_min):
        # Use TTS to say "Starting timer for X minutes"
        type_of_timer = "work" if not self.work else "break"
        tts = gTTS(text="Starting " + type_of_timer + " timer for "+str(duration_min)+" minutes", lang='en')
        tts.save(constants.TIMER_SOUND_FILEPATH)

        pygame.init()
        pygame.mixer.init()
        pygame.mixer.music.load(constants.TIMER_SOUND_FILEPATH)
        pygame.mixer.music.play()
        time.sleep(3)

    def start_timer(self):
        cur_time = time.time()
        if self.work:
            self.target_time = cur_time+25*60
            self.work_stage += 1
            self.work = False
            duration_min = 25
            GPIO.output(constants.LED_PIN_RED, True)
        else:
            if self.break_stage == 3:
                self.target_time = cur_time+15*60
                duration_min = 15
                self.break_stage = 0
                self.work_stage = 0
            else:
                self.target_time = cur_time+5*60
                duration_min = 5
                self.break_stage +=1
            GPIO.output(constants.LED_PIN_BLUE, True)
            self.work = True
        self.timer_in_progress = True
        self.play_timer_start_sound(duration_min)


    # End the timer and turn the light off
    def end_timer(self):
        
        self.target_time = 0
        # Play alarm sound
        pygame.init()
        pygame.mixer.init()
        pygame.mixer.music.load(constants.ALARM_SOUND_FILEPATH)
        pygame.mixer.music.play()
        time.sleep(1)
        pygame.mixer.music.play()
        time.sleep(1)
        pygame.mixer.music.play()
        time.sleep(1)
        # Turn off led
        GPIO.output(constants.LED_PIN_RED, False)
        GPIO.output(constants.LED_PIN_BLUE, False)
        self.timer_in_progress = False


# Perform duck functions
duck = Duck()
## Quack at startup
pygame.init()
pygame.mixer.init()
pygame.mixer.music.load(constants.ALARM_SOUND_FILEPATH)
pygame.mixer.music.play()
time.sleep(1)
gauth = GoogleAuth() 
drive = GoogleDrive(gauth)

while True:
    if duck.timer_in_progress and duck.target_time <= time.time():
        duck.end_timer()

    if GPIO.input(constants.TOUCH_SENSOR_RECORD_PIN):  # When the touch sensor is activated
        time.sleep(0.5)
        if GPIO.input(constants.TOUCH_SENSOR_TIMER_PIN):
            duck.toggle_chat()
        else:
            duck.toggle_recording()
    
    if GPIO.input(constants.TOUCH_SENSOR_TIMER_PIN):
        time.sleep(0.5)
        if GPIO.input(constants.TOUCH_SENSOR_RECORD_PIN):
            duck.toggle_chat()
        else:
            if duck.timer_in_progress:
                duck.end_timer()
            else:
                duck.start_timer()




