



import os
import openai
import os
import pygame
import tempfile
import time
from gtts import gTTS
import speech_recognition as sr
from dotenv import load_dotenv
import os




from openai import OpenAI

#pull api key from .env file please create a .env file and add your api key
client = OpenAI(api_key=os.getenv("OPEN_API_KEY"))
load_dotenv()

# create an environment variable called OPENAI_API_KEY and set it to your key
openai.api_key = os.getenv("OPEN_API_KEY")


#chat with the local model
def chat_lm(prompt):
    response = client.chat.completions.create(
        model="gpt-4-0613",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
    )
    return response.choices[0].message

#speaks the text
def speak(text):
    tts = gTTS(text=text, lang='en')
    temp = tempfile.NamedTemporaryFile(delete=False)
    temp.close()
    tts.save(temp.name)

    pygame.mixer.init()
    pygame.mixer.music.load(temp.name)
    pygame.mixer.music.play()
    print(text)
    while pygame.mixer.music.get_busy(): 
        pygame.time.Clock().tick(10)
    pygame.mixer.quit()


#listens to microphone
def listen():
    #create recognizer
    r = sr.Recognizer()
    #what microphone to use
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        print("Listening...")
        speak("Listening...")
        r.pause_threshold = 1
        #listen to microphone
        audio = r.listen(source)
        text = ''
    try:
        text = r.recognize_google(audio)
        
    except sr.RequestError as re:
        print(re)
        print("Sorry, I encountered an error. Please try again.")
    except sr.UnknownValueError as uve:
        print(uve)
        print("Sorry, I couldn't understand. Please try again.")
    except sr.WaitTimeoutError as wte:
        print(wte)
        print("Sorry, the operation timed out. Please try again.")
    text = text.lower()
    return text


#main loop
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
        #output response
        print(response)
        speak(response.content)
 