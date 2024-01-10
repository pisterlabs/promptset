import wikipedia
import datetime
import webbrowser
import os
import speech_recognition as sr
from AppKit import NSSpeechSynthesizer
#import openai_secret_manager
import openai
import re

engine = NSSpeechSynthesizer.alloc().init()

def speak(audio):
    engine.startSpeakingString_(audio)

def wishMe():
    hour = datetime.datetime.now().hour
    print(hour)
    if 0 <= hour < 12:
        speak("Good Morning!")
    elif 12 <= hour <= 18:
        print("e")
        speak("Good Afternoon!")
    else:
        speak("Good Evening!")
    speak("I am Jarvis, your virtual assistant. Please tell me how may I help you?")

def takeCommand():
    # It takes microphone input from the user and return string output
    r=sr.Recognizer()
    with sr.Microphone() as source:
        print('Listening...')
        r.pause_threshold = 1#use control + click to get info about pause_threshold
        audio = r.listen(source)

    try:
        print("Recognizing...")
        query =r.recognize_google(audio, language='en-in') 
        print(f"User said {query}\n")

    except Exception as e:
        #print(e)
        print('Say that again please....')
        return "None"#not the python None we return a string here 
    return query

def gpt_query(key,input_prompt):
  openai.api_key=key
  response = openai.Completion.create(model="text-davinci-003", prompt=input_prompt, temperature=0, max_tokens=60)
  print(response['choices'][0]['text'])
  speak(response['choices'][0]['text'])
  return 


if __name__ == "__main__":
    wishMe() 
    while True:
        query = takeCommand().lower()
        if 'wikipedia' in query:
            speak('Searching Wikipedia...')
            query = query.replace("wikipedia","")
            results= wikipedia.summary(query, sentences=1)
            speak("According to wikipedia")
            print(results)
            speak(results)
        elif 'open youtube' in query:
            webbrowser.open("youtube.com")
        elif 'open google' in query:
            webbrowser.open("google.com")
        elif 'open stack overflow' in query:
            webbrowser.open("stackoverflow.com")
        elif 'play music' in query:
            music_dir = 'D:\\Ai\\songs'
            songs = os.listdir(music_dir)
            print(songs)
            os.startfile(os.path.join(music_dir,songs[6]))#can also use random module from 0 to l-1
        elif 'the time' in query:
            strTime = datetime.datetime.now().strftime("%H:%M:%S")
            speak(f"The time is {strTime}")
        elif 'open code' in query:
            codePath = "open -a 'Visual Studio Code'"
            os.system(codePath)
        elif 'exit' in query:
            exit(1)
        elif 'camera' in query:
            import camera_timer
        else:
            print(query)
            key="<Your-openai-key-here>"
            gpt_query(key,query)
	        # message = query_gpt(query)
         #    speak(f"Here's what I found on that: {message}")
         #    print(f"Here's what I found on that: {message}")

