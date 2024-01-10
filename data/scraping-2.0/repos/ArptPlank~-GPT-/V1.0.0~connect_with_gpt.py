import os

from openai import OpenAI, api_key
import configparser
import threading
import time
import pygame
class gpt():
    def __init__(self):
        config = configparser.ConfigParser()
        config.read('config.ini')
        api_key = config.get("Settings", "openai_api_key")
        self.client = OpenAI(api_key=api_key)
        self.model_for_chat = "gpt-3.5-turbo"
        self.delta = []
        self.is_chat = False
        self.is_finish = False
        self.sound_path = []
        self.index = 0
        pygame.init()


    def split(self,text):
        _text = []
        text = text.split("\n")
        for t in text:
            if "用户：" in t:
                _text.append({"role":"user","content":t.replace("用户：","")})
            elif "GPT：" in t:
                _text.append({"role":"system","content":t.replace("GPT：","")})
            else:
                pass
        return _text

    def chat(self,text):
        self.is_chat = True
        text = text.replace("\n","")
        text = self.split(text)
        response = self.client.chat.completions.create(
            model=self.model_for_chat,
            messages=text,
            max_tokens=600,
            stream=True,
        )
        t = threading.Thread(target=self.get_response,args=(response,))
        t.start()


    def get_text_of_sound(self,sound_path):
        audio_file = open(sound_path, 'rb')
        transcript = self.client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            prompt="ZyntriQix, Digique Plus, CynapseFive, VortiQore V8, EchoNix Array, OrbitalLink Seven, DigiFractal Matrix, PULSE, RAPT, B.R.I.C.K., Q.U.A.R.T.Z., F.L.I.N.T."
        )
        return transcript.text

    def get_response(self,response):
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                self.delta.append(chunk.choices[0].delta.content)
        self.is_finish = True

    def get_real_sound(self,text,index):
        sound_path = f"output_sound/{time.time()}.mp3."
        response = self.client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text,
        )
        response.stream_to_file(sound_path)
        while True:
            if index == self.index:
                break
        pygame.mixer.init()
        pygame.mixer.music.load(sound_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pass
        self.index += 1
        #os.remove(sound_path)


if __name__ == '__main__':
    gpt = gpt()
    print(gpt.get_text_of_sound("mp3/1702134838.wav"))
    # print(gpt.chat("用户：你好\nGPT：你��
