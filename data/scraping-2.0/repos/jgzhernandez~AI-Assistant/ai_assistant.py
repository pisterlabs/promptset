import torch
import speech_recognition as sr
import openai
import os
from gtts import gTTS
import sounddevice as sd
import soundfile as sf
import json


DEVICE = torch.device('gpu' if torch.cuda.is_available() else 'cpu')
CHATBOT_RESPONSES = []


class Assistant:
    
    try:
        with open('api.json') as api:
            api_key = json.load(api)
            openai.api_key = api_key['api_key']
    except FileNotFoundError:
        pass

    def __init__(self):

        self.is_on = False
        self.is_transcribing = False
        self.is_listening = False
        self.is_closing = False
        self.recognizer = sr.Recognizer()

    def chatbot(self, prompt):
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=1024,
            n=1,
            temperature=0.5,
        )
        CHATBOT_RESPONSES.append(response.choices[0].text)

    def tts(self, text):
        audio = gTTS(text=text, lang="en", slow=False)
        audio.save("audio.wav")
        # print("Playing audio file")
        # Extract data and sampling rate from file
        data, fs = sf.read("audio.wav", dtype='float32')
        sd.play(data, fs)
        sd.wait()  # Wait until file is done playing
        os.remove("audio.wav")

    def voice_recognize(self):
        with sr.Microphone(sample_rate=16000) as source:
            self.recognizer.adjust_for_ambient_noise(source)
            while self.is_on:
                audio = self.recognizer.listen(source)
                try:
                    transcription = self.recognizer.recognize_google(audio)
                    transcription = transcription.lower()
                    if transcription in ["hey mom", "hi mom", "hello mom"]:
                        self.is_listening = True
                    elif transcription in ["thank you mom", "thanks mom"]:
                        self.is_listening = False
                        self.is_on = True
                    elif transcription in ["goodbye mom", "bye mom"]:
                        self.is_closing = True
                        self.is_on = False
                    elif self.is_listening and transcription:
                        self.is_transcribing = True
                        self.chatbot(transcription)
                        self.tts(CHATBOT_RESPONSES[-1].strip())
                        CHATBOT_RESPONSES.pop(-1)
                        self.is_transcribing = False
                except sr.UnknownValueError:
                    pass

