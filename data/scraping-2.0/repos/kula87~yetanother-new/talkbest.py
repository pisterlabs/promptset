import datetime
import os
import re
import requests
import sounddevice as sd
import soundfile as sf
import numpy as np
from colorama import Fore, Style, init
from pydub import AudioSegment
from pydub.playback import play
import openai

init()

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()
        
class ChatBot:
    def __init__(self, api_key, elapikey, voice_id):
        self.api_key = api_key
        self.elapikey = elapikey
        self.voice_id = voice_id
        self.conversation = []
        
    def add_to_conversation(self, role, content):
        self.conversation.append({"role": role, "content": content})

    def chat(self, chatbot, user_input, temperature=0.9, frequency_penalty=0.2, presence_penalty=0):
        openai.api_key = self.api_key
        self.add_to_conversation("user", user_input)
        messages_input = self.conversation.copy()
        prompt = [{"role": "system", "content": chatbot}]
        messages_input.insert(0, prompt[0])
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            temperature=temperature,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            messages=messages_input)
        chat_response = completion['choices'][0]['message']['content']
        self.add_to_conversation("assistant", chat_response)
        return chat_response

    def text_to_speech(self, text):
        url = f'https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}'
        headers = {
            'Accept': 'audio/mpeg',
            'xi-api-key': self.elapikey,
            'Content-Type': 'application/json'
        }
        data = {
            'text': text,
            'model_id': 'eleven_monolingual_v1',
            'voice_settings': {
                'stability': 0.6,
                'similarity_boost': 0.85
            }
        }
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            with open('output.mp3', 'wb') as f:
                f.write(response.content)
            audio = AudioSegment.from_mp3('output.mp3')
            play(audio)
        else:
            print('Error:', response.text)

    def print_colored(self, agent, text):
        agent_colors = {
            "Julie:": Fore.YELLOW,
        }
        color = agent_colors.get(agent, "")
        print(color + f"{agent}: {text}" + Style.RESET_ALL, end="")
    
    def record_and_transcribe(self, duration=8, fs=44100):
        print('Recording...')
        input_device_info = sd.query_devices(None, 'input')
        channels = input_device_info['max_input_channels']  # Get the maximum number of input channels
        myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=channels)
        sd.wait()
        print('Recording complete.')
        filename = 'myrecording.wav'
        sf.write(filename, myrecording, fs)
        with open(filename, "rb") as file:
            openai.api_key = self.api_key
            result = openai.Audio.transcribe("whisper-1", file)
        transcription = result['text']
        return transcription

if __name__ == "__main__":
    bot = ChatBot(api_key=open_file('openaiapikey2.txt'), elapikey=open_file('elabapikey.txt'), voice_id='21m00Tcm4TlvDq8ikWAM')

    chatbot1 = open_file('chatbot1.txt')

    while True:
        user_message = bot.record_and_transcribe()
        response = bot.chat(chatbot1, user_message)
        bot.print_colored("Julie:", f"{response}\n\n")
        user_message_without_generate_image = re.sub(r'(Response:|Narration:|Image: generate_image:.*|)', '', response).strip()
        bot.text_to_speech(user_message_without_generate_image)
