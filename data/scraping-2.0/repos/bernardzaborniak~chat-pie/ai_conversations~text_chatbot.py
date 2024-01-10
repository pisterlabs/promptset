from datetime import datetime
import os
import threading
import time
import speech_recognition as sr
from openai import OpenAI
from gtts import gTTS
import dynamic_recorder as dynamic_recorder

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

#accent could be
    #'com.au'
    #'co.uk'
    #'us'
    #'ca'
    #'co.in'
    #'ie
    #'co.za

class TextChatbot:
   
    def __init__(self, is_initiator, chat_character_prompt, accent):
        # Decides whether this chatbot starts a conversation first
        self.is_initiator = is_initiator
        # This prompt is used to shape the character of the chatbot
        self.chat_character_prompt = chat_character_prompt
        self.chat_history = []
        self.waiting_for_chat_gpt = False
        self.chat_gpt_response = ""
        self.chat_history = []
        self.accent = accent
        self.response_number = 0

        self.OUTPUT_FOLDER = os.path.join(os.getcwd(), "outputs")
        if os.path.exists(self.OUTPUT_FOLDER):
            import shutil
            shutil.rmtree(self.OUTPUT_FOLDER)
        os.makedirs(self.OUTPUT_FOLDER)


    def play_text(self, text):
        print("answer: " + text)
        with open(os.path.join(self.OUTPUT_FOLDER, f'output{self.response_number}.txt'), 'w') as f:
            f.write(text)
        self.response_number = self.response_number + 1


    def get_chat_gpt_response_threaded(self, input_text):
        self.chat_history.append({"role": "user", "content": input_text})
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", messages=self.chat_history
            #model="gpt-4", messages=self.chat_history
        )
        self.chat_gpt_response = response.choices[0].message.content
        self.chat_history.append({"role": "assistant", "content": self.chat_gpt_response})
        self.waiting_for_chat_gpt = False
        print("chat gpt finished: " + self.chat_gpt_response)


    def main(self):
        self.waiting_for_chat_gpt = True
        print('set up prompt is: ' + self.chat_character_prompt)
        threading.Thread(target=self.get_chat_gpt_response_threaded(self.chat_character_prompt)).start()

        print("initiating first prompt")
        print("waiting_for_chat_gpt")
        while self.waiting_for_chat_gpt:
            time.sleep(1)
            print("waiting for chat gpt... this can take some time...")
        
        if self.is_initiator:
            self.play_text(self.chat_gpt_response)
        else:
            print(self.chat_gpt_response)

        while True:        
            try:
                with open(os.path.join(self.OUTPUT_FOLDER, f'output{self.response_number}.txt')) as f:
                    input_text = f.read()
                print("input text read")
                self.response_number = self.response_number + 1

            except FileNotFoundError:
                time.sleep(5)
                continue

            print("input text: " + input_text)
            if input_text == '' or input_text == "Thank you.":
                continue

            self.waiting_for_chat_gpt = True
            threading.Thread(target=self.get_chat_gpt_response_threaded(input_text)).start()

            print("chat gpt time")
            print(self.waiting_for_chat_gpt)
            while self.waiting_for_chat_gpt:
                time.sleep(1)
                print("waiting for chat gpt... this can take some time...")

            self.play_text(self.chat_gpt_response)
