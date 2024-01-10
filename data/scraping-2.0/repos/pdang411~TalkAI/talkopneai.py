
import os
import openai
import pyttsx3
import speech_recognition as sr
from dotenv import load_dotenv
import os

from openai import OpenAI

#pull api key from .env file please create a .env file and add your api key
client = OpenAI(api_key=os.getenv("OPEN_API_KEY"))
load_dotenv()

# create an environment variable called OPENAI_API_KEY and set it to your key
openai.api_key = os.getenv("OPEN_API_KEY")



#chat with the local model go to openai chose your model to speak to and copy the model name
def chat_lm(prompt):
    response = client.chat.completions.create(
        model="gpt-4-0613",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

#speaks the text
def speak(text):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)
    voicespeed = 140
    engine.setProperty('rate', voicespeed) 
    engine.say(text)
    engine.runAndWait()


#listens to microphone
def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        speak("Listening...")
        audio = r.listen(source)
        try:
            return r.recognize_google(audio)
        except:
            return ""


#Main loop
if __name__ == "__main__":
    while True:
        human_input = listen()
        
        if not human_input:
            print("I didn't catch that. Could you please repeat?")
            speak("I didn't catch that. Could you please repeat?")
            continue

        if human_input.lower() in [ "quit", "exit", "stop", "bye", "goodbye"]:
            break
        
        response = chat_lm(human_input)
        #output from AI
        print("AI: " + response)
        speak(response.content)
