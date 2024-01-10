import openai
import pyttsx3
import speech_recognition as sr
import time
from app import *
#from Speech import speak

engine = pyttsx3.init()

def speak(message):
    engine = pyttsx3.init()
    rate = engine.getProperty('rate')
    print(rate) 
    engine.setProperty('rate', 170)     
    # engine.setProperty('voice', voices[1].id)
    volume = engine.getProperty('volume')
    print(volume)
    # voice = engine.getProperty('voice')
    # engine.setProperty('voices', voice[1].id)
    #print(voice)
    engine.say(message) 
    engine.runAndWait()

def printsometext(audio):
    recognizer = sr.Recognizer()
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        print("Could not understand audio")
    except sr.RequestError as e:
        print(f"Could not request results; {e}")


def Recognition_Text():
    recognizer = sr.Recognizer()
    print("Say Hello do me something...")
    with sr.Microphone() as source:
        print("Listening....")
        audio = recognizer.listen(source)
        #print("listened")
        try:
            #print("entered try catch loop")
            recognized_text = printsometext(audio) 
            print(recognized_text)
            #print("\n entering into testing if hey baldev was found")
            data = recognized_text.lower()
            print(data)
            if "hello" in data:
                # if recognized_text:
                    print(f"You said: {recognized_text}")
                    text = data.replace("hello","").strip()
                    #print(f"text removing hey baldev : {text}")
                    text_data = Generative_response(text)
                    print(f"Chatgpt response: {text_data}")
                    speak(text_data)
        except Exception as e:
            print("An error has occured : {}".format(e)) 
        
        # Save the recognized text to a file
        with open("recognized_text.txt", "w") as f:
            f.write(recognized_text)

if __name__ == "__main__":
    Recognition_Text()