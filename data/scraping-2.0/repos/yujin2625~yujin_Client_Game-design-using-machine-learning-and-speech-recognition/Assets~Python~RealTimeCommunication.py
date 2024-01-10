import speech_recognition as sr
import sys
import io
import os
import pyaudio
import openai
from google.cloud import translate_v2 as translate
from google.cloud import speech_v1p1beta1 as speech

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:\capstonestt-382711-aa7b9f69e752.json"
r = sr.Recognizer()
translate_client = translate.Client()


print("1초간 소음측정중..")

        
with sr.Microphone() as source :
    r.adjust_for_ambient_noise(source)




def Speak() :

    with sr.Microphone() as source :        
        print("이야기하세요..")

        audio_data = r.listen(source)

        print("인식중..")

        try:
            text = r.recognize_google(audio_data, language = 'ko-KR')
            print(text)

            result = translate_client.translate(text, target_language='en')
            translated_text = result['translatedText']
            print("번역 결과:", translated_text)

            result1 = translate_client.translate(translated_text, target_language='ko-KR')
            last = result1['translatedText']

            print("마지막 결과:", last)

        
        except sr.UnknownValueError:
            text = ""
            print("음성이 인식이 안됐어요")

 
while True :
    Speak()
