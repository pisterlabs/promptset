import os
import sys
import speech_recognition as speech
import pyttsx3
import pywhatkit
import playsound
from datetime import date
import time
import pynput
from pynput import keyboard
from pynput.keyboard import Key, Controller
import datetime
import wikipedia
import pyjokes
import openai


class Manju:
    """
    Daisy is a class that provides a simple interface to interact with the user using speech recognition and Text-to-Speech.
    The class uses several libraries such as speech_recognition, pyttsx3, pywhatkit, datetime, time, pynput, wikipedia, pyjokes.
    """

    def __init__(self) -> None:
        """
        The constructor method that initializes the listener, engine, and voices properties when an instance of the class is created.
        """
        self.listener = speech.Recognizer()
        self.engine = pyttsx3.init()
        self.responseList = []
        self.authentication = ""
        # self.voices = self.engine.getProperty('voices')
        # print(self.voices)
        # print(self.voices)
        # for voice in self.voices:
        #     self.engine.setProperty('voice', voice.id)
        #     self.engine.setProperty('voice', self.voices[1].id)

    def talk(self, text):
        """
        This method takes a single argument, a text string, and uses the `pyttsx3` library to speak the text.
        """
        self.engine.say(text)
        self.engine.runAndWait()

    def take_command(self):
        """
        This method listens to the microphone and uses the Google Speech Recognition service to recognize the audio.
        If the audio is not understood, it returns an empty string.
        """
        print("Listening")
        try:
            with speech.Microphone() as source:
                audio = self.listener.listen(source, timeout=8, phrase_time_limit=8)
                command = ""
                command = self.listener.recognize_google(audio)
                command = command.lower()
                if 'manju' in command:
                    command = command.replace('manju', '')
                    self.executioner(command)
                    print(command)
        except speech.UnknownValueError as e:
            print("Could not understand audio")
        return command

    def send_whatsapp_message(self, name: str):
        """
        This method takes a single argument, a name, and uses the `pywhatkit` library to send a WhatsApp message to the number associated with the given name.
        """
        if "john" in name:
            pywhatkit.sendwhatmsg_instantly("+1787242700", "Hello Sir")

    def send_whatsapp_image(self, phone_number: str, image: str):
        """
        This method takes two arguments, phone number and image, and uses the `pywhatkit` library to send a WhatsApp image to the given phone number.
        """
        pywhatkit.sendwhats_image(phone_number, image)

    def play_media(self, media: str):
        """
        This method takes a single argument, media, and uses the `pywhatkit` library to play the media on youtube.
        """
        self.talk('playing ' + media + 'on YouTube')
        pywhatkit.playonyt(media)

    def current_time(self):
        """
        This method tells the current time using `datetime` library.
        """
        currentTime = datetime.datetime.now().strftime('%I:%M %p')
        self.talk('Current time is ' + currentTime)

    def send_message(self, phone_number: str):
        """
        This method takes phone number as an argument and uses `os` and `pynput` library to open message application and send message to the given phone number.
        """
        os.system("open -a Messages")
        keyboard = Controller()
        keyboard.type("Hi beautiful, how are you doing?")
        keyboard.press(Key.enter)
    
    def askChatGPT(self, command: str):
        # openai.api_key = os.getenv("OPENAI_API_KEY")
        openai.api_key = self.authentication


        chatGPTResponse = openai.Completion.create(
            model="text-davinci-003",
            prompt= command,
            temperature = 0)
        self.responseList.append(chatGPTResponse["choices"][0]["text"])
        print(self.responseList)
        return self.responseList

    def executioner(self, command):
        actions = {
            "play": self.play_media,
            "message": lambda: self.send_whatsapp_message("john"),
            "image": lambda: self.send_whatsapp_image('phone number', 'image path'),
            "time": self.current_time,
            "text": lambda: self.send_message('phone number')
        }
        for action, func in actions.items():
            if action in command:
                func()
                break
        else:
            return self.askChatGPT(command)
        return ["Sorry couldn't find an answer"]