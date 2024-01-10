import pyttsx3
import datetime
import speech_recognition as sr
import wikipedia
import webbrowser
import os
import smtplib
import openai
from time import sleep
import pyautogui

engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
#print(voices[1].id)
engine.setProperty('voices' , voices[0].id)


def speak (audio):
    "with this jarvis speaks"
    engine.say(audio)
    engine.runAndWait()

def wishMe():
    hour =int(datetime.datetime.now().hour)
    if hour>=0 and hour<12:
        speak("good morning")

    elif hour>=12 and hour<=18:
        speak("good afternoon")

    else:
        speak("good evening")
    
    speak("i am Jarvis sir how may i help you")

def takeCommand():
    "it takes microphone input from the user and gives string output"

    r=sr.Recognizer()
    with sr.Microphone() as source :
        print("listening...")
        r.pause_threshold=1
        audio = r.listen(source)

    try:
        print("Recognizing..")
        query = r.recognize_google(audio,language='en-in')
        print(f"User said: {query}\n")

    except Exception as e:
        print(e)#to print the exception
        print("say that again please")
        return "NONE"
       
    return query
        

def sendEmail(to,content):
    server = smtplib.SMTP('smtp.gmail.com',587)#587 is the port
    server.ehlo()
    server.start.ls()
    server.login('your gmail','your gmail password')
    server.sendmail('receiver gmail',to,content)
    server.close()
    

def GPT(query):
     openai.api_key="open ai api key"
     model_engine="text-davinci-003"#engine used is davinci may at your time it may be discarded
     prompt=query

     completion = openai.Completion.create(
     engine=model_engine,
     prompt=prompt,
     max_tokens=1024,
     n=1,
     stop=None,
     temperature=0.5
     )

     response =completion.choices[0].text
     print(response)
     return (response)   


if __name__ == "__main__":
    wishMe()
    while True :
       #query ='GPT indigo'
       query=takeCommand().lower()

       #logic for executing talks based on query
       if 'wikipedia' in query:
           speak('Searching Wikipedia...')
           query=query.replace("wikipedia","")
           results=wikipedia.summary(query,sentences=2,auto_suggest=False)##it was auto suggesting asia to as8a hence made 
           speak("According to wikipedia")
           print(results)
           speak(results)
           #break
       
       elif 'open youtube' in query:
           webbrowser.open("youtube.com")#how to open it on chrome
           #break
       
       elif '.com' in query:
           query=query.replace("jarvis"," ")
           query=query.replace("open"," ")
           query=query.strip()
           print(query)
           webbrowser.open(query)
           #break
       
       elif 'play offline music '  in query:#for offline music
           music_dir='dir path'
           songs=os.listdir(music_dir)
           print(songs)
           os.startfile(os.path.join(music_dir,songs[0]))#try using random module to play a song
           #break
            
       elif 'the time' in query:
           strTime=datetime.datetime.now().strftime("%H:%M%S")
           speak(f"the time is {strTime}")
           #break
       
       elif 'open code' in query :
           codePath = "set c drive path it is specifically for visual code studio it can be designed for others also"
           os.startfile(codePath)
           #break
       
       elif 'email' in query:#make dictionary use try and except#enable less secure apps to use this function
           try:
               speak("what should i say!")
               content = "hello send this trying to fugure out"
               to = "asc15.lko@gmail.com"
               sendEmail(to,content)
               speak("email has been sent!")
           except Exception as e:
               print (e)
               speak("not able to send email")
               #break
               
       elif 'GPT' in query :
           speak("According to open ai chatgpt")
           query =query.replace('GPT','')
           response=GPT(query)
           speak(response)
           #break
       
       elif 'listen song spotify' in query or 'play music' in query:
           speak("sir what song do you like to listen")
           song=takeCommand().lower()
           webbrowser.open(f'https://open.spotify.com/search/{song}')
           sleep(13)
           pyautogui.click(x=1055,y=617)
           speak("playing"+song)
           #break

           

           




       
       

