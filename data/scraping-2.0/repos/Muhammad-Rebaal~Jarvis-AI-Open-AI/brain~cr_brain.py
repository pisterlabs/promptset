fileopen = open("brain\API_Key.txt","r")
API = fileopen.read()
fileopen.close()

import openai
from dotenv import load_dotenv
import pyttsx3
import speech_recognition as sr
from googletrans import Translator

openai.api_key = API
load_dotenv()
completion = openai.Completion()

def reply(question, chat_log = None):
    with open("abc.txt","a+") as File_log:
        if chat_log is None:
            chat_log_template = ""
        else:
            chat_log_template = chat_log

        prompt = f"{chat_log_template} You : {question}\n Jarvis: "
        response = completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=0.7,
            max_tokens=60,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0    
        )
        answer = response.choices[0].text.strip()
        chat_log_template_update = chat_log_template + f"\n You: {question}\n Jarvis : {answer}"
        File_log.write(chat_log_template_update)
        return answer

engine = pyttsx3.init()

def speak(audio):
    engine.say(audio)
    engine.runAndWait()

def tc():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        r.pause_threshold = 1
        r.energy_threshold =3000 #--->There are other voices in the room use energy_threshold.
        audio = r.listen(source)
        

    try:
        print("Recognizing...")
        query = r.recognize_google(audio,language ='en-US')
        line = str(query)
        translate = Translator()
        result = translate.translate(line)
        data = result.text
        print(data)

    except Exception as e:
        print(e)
        speak("Sir,please say that again")
        return "None"
    return data


if __name__ == "__main__":
    # wishme()
    
    while True:
        data= tc().lower()
        speak(reply(data))
        if "stop" in data:
            speak("ok sir! I am going offline")
            exit()
        