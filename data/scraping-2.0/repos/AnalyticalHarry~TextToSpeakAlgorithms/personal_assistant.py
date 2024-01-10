import time
import openai
import pyttsx3
import threading
import ipywidgets as widgets
import speech_recognition as sr

from openai import OpenAI
from IPython.display import display

def speech(text):
        engine = pyttsx3.init() 

        """ RATE"""
        #current speaking rate details
        rate = engine.getProperty('rate')   
        print()
        #printing current voice rate
        print(f"Speed rate of speech: {rate}")  
        #setting up new voice rate
        engine.setProperty('rate', 180)     

        """VOLUME"""
        #current volume level (min=0 and max=1)
        volume = engine.getProperty('volume')  
        print()
        #current volume level
        print(f"Volume of speech: {volume}")   
        #setting up volume level  between 0 and 1
        engine.setProperty('volume',1.0)    
        engine.say(text)
        engine.runAndWait()
        engine.stop()

        """Saving Voice to a file"""
        engine.save_to_file(text, 'test.mp3')
        engine.runAndWait()
    

recognizer = sr.Recognizer()

with sr.Microphone() as source:
    print("Listening for voice input...")

    try:
        audio = recognizer.listen(source, timeout=10)
        print("Audio input received. Recognizing...")

        # Recognize the speech
        text = recognizer.recognize_google(audio)
        print("Recognized Text: " + text)

    except sr.WaitTimeoutError:
        print("No speech input detected. Exiting.")


#function called chat_gpt that takes one argument recognized_text.
def chat_gpt(recognized_text):  
    #API_KEY variable to a specific API key.
    API_KEY = openai.api_key = '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'  
    #client object to interact with the OpenAI API.
    client = OpenAI(  
         #pass the API key to the client.
        api_key=API_KEY 
    )
    #recognized_text argument in the prompt variable.
    prompt = text
    #client to create a chat completion.
    chat_completion = client.chat.completions.create(  
        messages=[
            {
                #"user" for the message.
                "role": "user",  
                #recognized_text.
                "content": prompt  
            },
        ],
        #LLM model 
        model="gpt-3.5-turbo"  
    )
    #the content of the completed chat response.
    text_to_speak = (chat_completion.choices[0].message.content)  
    #content of the response to the console.
    print(text_to_speak)  
    return speech(text_to_speak)  

chat_gpt(text)