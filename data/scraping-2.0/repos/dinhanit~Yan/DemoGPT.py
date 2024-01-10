import os
import openai
import pyttsx3
from gtts import gTTS
import speech_recognition as sr
import YanAPI,re

class GPT:
    def __init__(self,key,lang="en-US"):
        openai.api_key = key
        self.lang = lang
        no_content = 0
      
    def generate_chat_completion(self,prompt):
        model_engine = "text-davinci-003"
        max_tokens = 1024
        temperature = 0.7
        response = openai.Completion.create(
            engine=model_engine,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            n=1,
            stop=None,
            )
        message = response.choices[0].text.strip()
        message = re.sub(r"\n", " ", message)
        message = re.sub(r"\s+", " ", message).strip()
        return message

    def speech_to_text(self):
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("Speak something...")
            audio = r.listen(source)
        try:
            text = r.recognize_google(audio,language=self.lang)
            print("You said: ", text)
            return text
        except: 
            return ""

    def text_to_speech(self,text):
        
        tts = gTTS(text=text, lang=self.lang)
        tts.save('output.mp3')
        YanAPI.upload_media_music('output.mp3')
        YanAPI.sync_play_music('output.mp3')
        YanAPI.delete_media_music('output.mp3')
        
    def converse(self):
        prompt = "Chatbot: Tôi có thể giúp gì"
        while True:
            cache = prompt
            cache_split = cache.split('Chatbot:')
            text = cache_split[-1]
            print(text+'\n')
            self.text_to_speech(text)
            user_input = self.speech_to_text() 
            if user_input != '':
                if user_input.lower() in ["quit", "exit", "bye", "kết thúc",'tạm biệt']:
                    self.text_to_speech("Tạm biệt, cảm ơn bạn đã lắng nghe")
                    break
                print(user_input)
                prompt += "\nUser: " + user_input.strip()
                response = self.generate_chat_completion(prompt)
                answer = response.strip()
                prompt += "\nChatbot: " + answer
                last_user_message = ""
                matches = re.findall(r"User: (.+)", prompt)
                if matches:
                    last_user_message = matches[-1]
                response = response.replace("{last_user_message}", last_user_message)



with open('apikey.txt') as f:
    key = f.read()
gpt = GPT(key,lang="vi") #,lang="en-US"s
gpt.converse()