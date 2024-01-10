import openai
# from api_key import API_KEY

import pyttsx3 as tts
import speech_recognition as sr

openai.api_key = "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"

# Implemented GPT3 from https://beta.openai.com/ testing GPT3 for conversation. Create your API key and create a api_key.py file add API_KEY = "yourkey" to the file.

recognizer = sr.Recognizer()


def abriella_speak(text):
        engine = tts.init()
        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices[1].id)
        engine.setProperty('rate', 155)
        engine.say(text)
        engine.runAndWait()


conversation = ""
user_name = "David"

with sr.Microphone() as source:
# implement the openai api
    while True:
        
        recognizer.adjust_for_ambient_noise(source, duration=0.2)
        audio = recognizer.listen(source)
        said = ''
        try:
            user_input = recognizer.recognize_google(audio)
        except:
            continue

        prompt = user_name + ": " + user_input + "\n Abriella:"

        conversation += prompt

        response = openai.Completion.create(engine='text-davinci-003', prompt=conversation, max_tokens=70)
        text = response["choices"][0]["text"].replace("\n", "")
        text = text.split(user_name + ": ", 1)[0].split("Abriella: ", 1)[0]

        conversation += text + "\n"
    
        print("I'm listening... speak clearly into mic.")
        print(f"{user_name}: {user_input}")
        abriella_speak(text)

