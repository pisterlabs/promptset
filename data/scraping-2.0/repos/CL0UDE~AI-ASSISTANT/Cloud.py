from ast import main
import os
from socket import if_nameindex
import speech_recognition as sr
import wikipedia
import datetime
import smtplib
import webbrowser
import pyttsx3
import random
import smtplib
from AppOpener import open,close
import openai
from config import apikey

engine = pyttsx3.init('sapi5')
voices = engine.getProperty("voices")
engine.setProperty('voice',voices[0].id)

mail_dict = {
    "Siddharth" : "someone@gmail.com",
    "Ayush" : "someone@gmail.com",
    "Pratap" : "someone@gmail.com"
}



def speak(audio):
    engine.say(audio)
    engine.runAndWait()


def greet():
    time = datetime.datetime.now().hour
    if (time>=0 and time<12):
        speak('Good morning')
    elif (time>=12 and time<5):
        speak('Good Afternoon')
    else:
        speak("Good evening!")

    speak("I am Cloud")

def command():
    input = sr.Recognizer()
    with sr.Microphone() as source:
        print('Listening...')
        input.pause_threshold = 1
        audio = input.listen(source)
    
    try:
        query = input.recognize_google(audio,language='en-in')
        print(f"input = {query}\n")

    except Exception as e:
        #print(e)
        speak("couldn't recognize")
        return "None"
    return query

def Ai(prompt):
    openai.api_key = apikey
    text = (f"Response for prompt :{prompt}\n*****\n\n")

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    print(response['choices'][0]['text'])
    ##text = text + (response['choices'][0]['text'])    
    ##if not os.path.exists("Openai"):
        ##os.mkdir("Openai")

    ##with open(f"Openai/{prompt[0:30]}",'w') as f:
        ##f.write(response)
    ##f.close()

def Email(to,content):
    server = smtplib.SMTP("smtp-mail.outlook.com", 587)
    server.ehlo()
    server.starttls()
    server.login("sender@outlook.com","password-example")
    server.sendmail("sender@outlook.com",to,content)
    server.close()


if __name__=='__main__':
    greet()
    query = command().lower()
    
    if "who is" in query:
        print("searching wikipedia...")
        speak("searching wikipedia")
        result = wikipedia.summary(query,sentences = 2)
        print(result)
        speak("According to wikipedia")
        speak(result)

    elif "open google" in query:
        webbrowser.open("google.com")
        
    elif "open youtube" in query:
        webbrowser.open("youtube.com")

    elif "open chrome" in query:
        chromePath = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"
        os.startfile(chromePath)

    elif "open code" in query:
        codePath = "C:\\Users\\siddh\\AppData\\Local\\Programs\\Microsoft VS Code\\Code.exe"
        os.startfile(codePath)

    elif "time" in query:
        print(datetime.datetime.now())
        speak(f"current time is {datetime.datetime.now()}")

    elif "play music" in query:
        song_list = "D:\\songs"
        music = os.listdir(song_list)
        i = random.randint(0,len(music)-1)
        os.startfile(os.path.join(song_list,music[i]))

    elif "open spotify" in query:
        webbrowser.open("spotify.com")

    elif "send a mail" in query:
        try:
            print("Who do u want to send the mail to?")
            speak("Who do u want to send the mail to?")
            print(mail_dict)
            to = (mail_dict[command()])
            print("what should I Send?")
            speak("What should I send?")
            content = command()
            Email(to,content)
            print(f"user said:{content}")
            print("The email was sent.")
            speak("The email was sent.")

        except Exception as e:
          print(e)
          print("sorry! the mail was not sent")
    
    elif "open" in query:
        try:
            app_name = (query[5:len(query)])
            open(app_name)

        except Exception as e:
            print(e)
            print("sorry couldn't find the app on your system.")

    elif "close" in query:
        try:
            app_name = (query[5:len(query)])
            close(app_name)
        except Exception as e:
            print("Couldn't close the program")

    elif "write" in query:
        Ai(prompt=query)
    
    elif "quit" in query:
        speak("Thank you")