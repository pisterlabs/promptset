import pyttsx3
import speech_recognition as sr
import datetime
import wikipedia
import webbrowser
import os
import smtplib
import requests
import wolframalpha
import subprocess
import pyjokes
import time
import google.generativeai
from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm
from bs4 import BeautifulSoup
from PyInquirer import Separator, prompt
import markdown

questions = [
    {
        'type': 'list',
        'name': 'assistant_type',
        'message': 'Select an assistant type:',
        'choices': [
            'Mike',
            'Annie',
            Separator(),
            {
                'name': 'Exit',
                'value': 'exit'
            }
        ]
    }
]
model_id="models/gemini-pro"
API = input("Enter your Google API Key: ")
llm=GooglePalm(google_api_key=API)
llm.temperature=0.7

print("DonutAI PREVIEW v2")
print("Built On Donut Assistant")
print("Gautham Nair")
print("More AI Features Coming Soon")

assistanttype = prompt(questions)
if assistanttype['assistant_type'] == "Mike":
    engine = pyttsx3.init()
    engine.setProperty('rate',150)
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)


    def speak(audio):                   
        engine.say(audio)
        engine.runAndWait()


    def wishMe():
        hour = int(datetime.datetime.now().hour)
        if hour>=0 and hour<12:
            print("Good Morning!")
            speak("Good Morning!")

        elif hour>=12 and hour<14:
            print("Good Afternoon!")
            speak("Good Afternoon!")

        else:
            print("Good Evening!")
            speak("Good Evening!")

        print("I am DonutAI Mike. Please tell me how may I help you?")
        speak("I am DonutAI Mike. Ask me anything!!!")

    def takeCommand():
        #It takes microphone input from the user and returns string output

        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("Listening...")
            speak("I'm Listening..!")
            r.pause_threshold = 1
            audio = r.listen(source)

        try:
            print("Recognizing...")    
            query = r.recognize_google(audio, language='en-in')
            print(f"User said: {query}\n")

        except Exception as e:
            # print(e)    
            print("Say that again please...")
            speak('I didnt hear anything, if you said anything please speak loud and clear')
            return ""
        return query

    def sendEmail(to, content):
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.ehlo()
        server.starttls()
        email = input("Enter your gmail username: ")
        psswrd = input("Enter yourn gmail password: ")
        try:
            server.login(email, psswrd)
            server.sendmail(email, to, content)
            server.close()
            print("E-mail kas been sent")
            speak("E-Mail has been sent")
        except Exception as e:
            print("An unexpected error occurred, did you try enabling less secure application in your Gmail Settings?")
            speak("An unexpected error occurred, did you try enabling less secure application in your Gmail Settings?")

    if __name__ == "__main__":
        wishMe()
        while True:
        # if 1:
            query = takeCommand().lower()

            # Logic for executing tasks based on query
            if 'wikipedia' in query:
                speak('Searching Wikipedia...')
                query = query.replace("wikipedia", "")
                results = wikipedia.summary(query, sentences=2)
                print("Generating Answers..!")
                speak("According to Wikipedia")
                print(results)
                speak(results)

            elif query == "tell me some jokes" or query == "tell some jokes" or query == "tell a joke" or query == "joke" or query == "jokes":
                My_joke = pyjokes.get_joke(language="en", category="neutral")
                print(My_joke)
                speak(My_joke)


            elif 'question' in query:
                speak('I can answer to computational and geographical questions, what question do you want to ask now')
                question=takeCommand()
                client = wolframalpha.Client('UL8UPY-4EHX5683WH')
                res = client.query(question)
                answer = next(res.results).text
                speak(answer)
                print(answer)

            elif "calculate" in query:

                app_id = "UL8UPY-4EHX5683WH"
                client = wolframalpha.Client(app_id)
                indx = query.lower().split().index('calculate')
                query = query.split()[indx + 1:]
                res = client.query(' '.join(query))
                answer = next(res.results).text
                print("The answer is " + answer)
                speak("The answer is " + answer)

            elif 'open youtube' in query:
                speak('OK, I will open YouTube in your default browser')
                webbrowser.open("youtube.com")

            elif 'open browser' in query:
                webbrowser.open("google.com")

            elif 'open google' in query or 'open Google' in query:
                speak('Opening Google in your default browser')
                webbrowser.open("google.com")

            elif 'open bing' in query:
                speak('Opening bing in your default browser')
                webbrowser.open("bing.com")

            elif 'send feedback' in query:
                speak('This will open Donut Support Website in your default browser, you can give feedback there!')
                webbrowser.open("Donutsupport.simdif.com")

            elif 'open stackoverflow' in query or 'open stack overflow' in query:
                speak('Opening StackOverflow in your default browser')
                webbrowser.open("stackoverflow.com")   


            elif 'play music' in query:
                try:
                    musidir = input("Enter directory address: ")
                    music_dir = musidir
                    songs = os.listdir(music_dir)
                    print(songs)    
                    os.startfile(os.path.join(music_dir, songs[0]))
                except:
                    speak("Sorry Friend!! I couldn't find the directory specified")

            elif 'time' in query:
                strTime = datetime.datetime.now().strftime("%H:%M:%S")
                print(strTime)
                speak(f"Friend, the time is {strTime}")

            elif 'text to speech' in query:
                text = input("Type: ")
                speak(text)

            elif 'when is your birthday' in query:
                print("1st March 2022")
                speak('I made my debut on 1st March 2022')

            elif 'your developers name' in query:
                print("Gautham Nair")
                speak("Gautham Nair")
            
            elif 'who developed you' in query:
                print("Gautham Nair")
                speak("Gautham Nair")

            elif 'what is your developers name' in query:
                print("Gautham Nair")
                speak("Gautham Nair")

            elif 'open code' in query:
                codePath = "code"
                os.startfile(codePath)

            elif 'what is your name' in query:
                speak('As I told you in the beginning, my name is DonutAI Mike')
                print("I am DonutAI Mike")

            elif 'who made you' in query:
                speak('Who made me??? Gautham Nair')
                speak('He is super genius')

            elif 'what do you eat' in query:
                speak("I dont't eat the food that humans eat, but i like to have bits and bytes")

            elif 'where do you live' in query:
                speak("I live in your computer")

            elif 'can you sing a song' in query:
                speak('Im noot good at singing, since i am a bot')
                speak('But since you asked me, i will sing it for you')
                speak("I will sing my favourite song")
                speak("The song is Michael Jackson's Smooth Criminal") 
                speak('''As he came into the window!!!!
                            Was the sound of a crescendo!!!
                            He came into her apartment!!!
                            He left the bloodstains on the carpet!!!
                            She ran underneath the table!!!
                            He could see she was unable!!!
                            So she ran into the bedroom!!!
                            She was struck down, it was her doom!!!
                            Annie, are you okay?
                            So, Annie, are you okay? Are you okay, Annie?
                            Annie, are you okay?
                            So, Annie, are you okay? Are you okay, Annie?
                            Annie, are you okay?
                            So, Annie, are you okay? Are you okay, Annie?
                            Annie, are you okay?
                            So, Annie, are you okay? Are you okay, Annie?
                            Annie, are you okay?
                            Will you tell us that you're okay?
                            There's a sound at the window
                            Then he struck you, a crescendo Annie
                            He came into your apartment
                            He left the bloodstains on the carpet
                            And then you ran into the bedroom
                            You were struck down
                            It was your doom
                            Annie, are you okay?
                            So, Annie, are you okay? Are you okay, Annie?
                            Annie, are you okay?
                            So, Annie, are you okay? Are you okay, Annie?
                            Annie, are you okay?
                            So, Annie, are you okay? Are you okay, Annie?
                            You've been hit by-
                            You've been hit by-
                            A smooth criminal
                            So they came in to the outway
                            It was Sunday, what a black day
                            Mouth-to-mouth resuscitation
                            Sounding heartbeats, intimidation
                            Annie, are you okay?
                            So, Annie, are you okay? Are you okay, Annie?
                            Annie, are you okay?
                            So, Annie, are you okay? Are you okay, Annie?
                            Annie, are you okay?
                            So, Annie, are you okay? Are you okay, Annie?
                            Annie, are you okay?
                            So, Annie, are you okay? Are you okay, Annie?
                            Annie, are you okay?
                            Will you tell us that you're okay?
                            There's a sound at the window
                            That he struck you a crescendo Annie
                            He came into your apartment
                            He left the bloodstains on the carpet
                            Then you ran into the bedroom
                            You were struck down
                            It was your doom
                            Annie, are you okay? So, Annie, are you okay?
                            Are you okay, Annie?
                            You've been hit by-
                            You've been struck by-
                            A smooth criminal
                            Okay, I want everybody to clear the area right now
                            Annie, are you okay? (I don't know)
                            Will you tell us, that you're okay? (I don't know)
                            There's a sound at the window (I don't know)
                            Then he struck you, a crescendo Annie (I don't know)
                            He came into your apartment (I don't know)
                            Left bloodstains on the carpet (I don't know why, baby)
                            And then you ran into the bedroom (help me)
                            You were struck down
                            It was your doom, Annie (dag gone it)
                            Annie, are you okay? (Dag gone it-baby)
                            Will you tell us that you're okay? (Dag gone it-baby)
                            There's a sound at the window (dag gone it-baby)
                            Then he struck you, a crescendo Annie
                            He came into your apartment (dag gone it)
                            Left bloodstains on the carpet (hoo, hoo, hoo)
                            And then you ran into the bedroom
                            You were struck down (dag gone it)
                            It was your doom Annie''')

            elif 'can i change your name' in query:
                print("Sorry Friend!")
                speak("Sorry Friend!, only my developers can change my name")

            elif 'do you know alexa' in query:
                speak("Yes, I know her, I want to be famous like her one day")

            elif 'do you know cortana' in query:
                speak("Yes, I know her, I want to be famous like her one day")

            elif 'do you know google assistant' in query:
                speak("Yes, I know her, I want to be famous like her one day")

            elif 'do you know siri' in query:
                speak("Yes, I know her, I want to be famous like her one day")

            elif 'do you know bixby' in query:
                speak("Yes, I know her, I want to be famous like her one day")

            elif 'who is your favourite artist' in query:
                print("Michael Jackson")
                speak('No doubt,, its michael jackson')

            elif 'exit' in query:
                print("Goodbye!!")
                speak('Goodbye!!, you can call me anytime')
                break

            elif 'email' in query:
                try:
                    useria = input("Email to whom?..Type it: ")
                    speak("What should I say?")
                    content = takeCommand()
                    to = useria    
                    sendEmail(to, content)
                except Exception as e:
                    print(e)
                    speak("Sorry my friend. I am not able to send this email")

            elif "log off" in query or "sign out" in query:
                speak("Ok , your pc will log off in 10 sec make sure you exit from all applications")
                subprocess.call(["shutdown", "/l"])
            
            else:
                if query == "":
                    print()
                else:
                    try:
                        prompt = [query]
                        llm_results= llm._generate(prompt)
                        res=llm_results.generations
                        print("Generating Answers...!")
                        print()
                        print(res[0][0].text)
                        speak(res[0][0].text)
                    except Exception as e:
                        print(e)
                        speak("Sorry, I could not generate an answer for that.!")

    time.sleep(3)

elif assistanttype['assistant_type'] == "Annie":
    engine = pyttsx3.init()
    engine.setProperty('rate',150)
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)


    def speak(audio):
        engine.say(audio)
        engine.runAndWait()


    def wishMe():
        hour = int(datetime.datetime.now().hour)
        if hour>=0 and hour<12:
            print("Good Morning!")
            speak("Good Morning!")

        elif hour>=12 and hour<18:
            print("Good Afternoon!")
            speak("Good Afternoon!")

        else:
            print("Good Evening!")
            speak("Good Evening!")

        print("I am DonutAI Annie. Please tell me how may I help you")
        speak("I am DonutAI Annie. Ask me anything!!!")

    def takeCommand():
        #It takes microphone input from the user and returns string output

        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("Listening...")
            speak("I'm Listening..!")
            r.pause_threshold = 1
            audio = r.listen(source)

        try:
            print("Recognizing...")    
            query = r.recognize_google(audio, language='en-in')
            print(f"User said: {query}\n")

        except Exception as e:
            # print(e)    
            print("Say that again please...")
            speak('I didnt hear anything, if you said anything please speak loud and clear')
            return ""
        return query

    def sendEmail(to, content):
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.ehlo()
        server.starttls()
        email = input("Enter your gmail username: ")
        psswrd = input("Enter yourn gmail password: ")
        try:
            server.login(email, psswrd)
            server.sendmail(email, to, content)
            server.close()
        except Exception as e:
            print("An unexpected error occurred, did you try enabling less secure application in your Gmail Settings?")
            speak("An unexpected error occurred, did you try enabling less secure application in your Gmail Settings?")

    if __name__ == "__main__":
        wishMe()
        while True:
        # if 1:
            query = takeCommand().lower()

            # Logic for executing tasks based on query
            if 'wikipedia' in query:
                speak('Searching Wikipedia...')
                query = query.replace("wikipedia", "")
                results = wikipedia.summary(query, sentences=2)
                speak("According to Wikipedia")
                print(results)
                speak(results)

            elif query == "tell me some jokes" or query == "tell some jokes" or query == "tell a joke" or query == "joke" or query == "jokes":
                My_joke = pyjokes.get_joke(language="en", category="neutral")
                print(My_joke)
                speak(My_joke)


            elif 'question' in query:
                speak('I can answer to computational and geographical questions  and what question do you want to ask now')
                question=takeCommand()
                client = wolframalpha.Client('UL8UPY-4EHX5683WH')
                res = client.query(question)
                answer = next(res.results).text
                speak(answer)
                print(answer)

            elif "calculate" in query:

                app_id = "UL8UPY-4EHX5683WH"
                client = wolframalpha.Client(app_id)
                indx = query.lower().split().index('calculate')
                query = query.split()[indx + 1:]
                res = client.query(' '.join(query))
                answer = next(res.results).text
                print("The answer is " + answer)
                speak("The answer is " + answer)

            elif 'open youtube' in query:
                speak('OK, I will open YouTube in your default browser')
                webbrowser.open("youtube.com")

            elif 'open browser' in query:
                webbrowser.open("google.com")

            elif 'open bing' in query:
                speak('Opening bing in your default browser')
                webbrowser.open("bing.com")

            elif 'send feedback' in query:
                speak('This will open Donut Support Website in your default browser, you can give feedback there!')
                webbrowser.open("Donutsupport.simdif.com")

            elif 'open google' in query or 'open Google' in query:
                speak('Opening google in your default browser')
                webbrowser.open("google.com")

            elif 'open stackoverflow' in query or 'open stack overflow' in query:
                speak('Opening StackOverflow in your default browser')
                webbrowser.open("stackoverflow.com")   


            elif 'play music' in query:
                try:
                    musidir = input("Enter directory address: ")
                    music_dir = musidir
                    songs = os.listdir(music_dir)
                    print(songs)    
                    os.startfile(os.path.join(music_dir, songs[0]))
                except:
                    speak("Sorry Friend!! I couldn't find the directory specified")

            elif 'time' in query:
                strTime = datetime.datetime.now().strftime("%H:%M:%S")
                print(strTime)
                speak(f"Friend, the time is {strTime}")

            elif 'text to speech' in query:
                text = input("Type: ")
                speak(text)

            elif 'when is your birthday' in query:
                print("1st March 2022")
                speak('I made my debut on 1st March 2022')

            elif 'your developers name' in query:
                print("Gautham Nair")
                speak("Gautham Nair")
            
            elif 'who developed you' in query:
                print("Gautham Nair")
                speak("Gautham Nair")

            elif 'what is your developers name' in query:
                print("Gautham Nair")
                speak("Gautham Nair")

            elif 'open code' in query:
                codePath = "code"
                os.startfile(codePath)

            elif 'what is your name' in query:
                speak('As I told you in the beginning, my name is DonutAI Annie')
                print("I am DonutAI Annie")

            elif 'who made you' in query:
                speak('Who made me??? Gautham nair')
                speak('He is a super genius')

            elif 'what do you eat' in query:
                speak("I dont't eat the food that humans eat, but i like to have bits and bytes")

            elif 'where do you live' in query:
                speak("I live in your computer")

            elif 'can you sing a song' in query:
                speak('Im noot good at singing, since i am a bot')
                speak('But since you asked me, i will sing it for you')
                speak("I will sing my favourite song")
                speak("This song has my name in it!!")
                speak("The song is Michael Jackson's Smooth Criminal") 
                speak('''As he came into the window!!!!
                            Was the sound of a crescendo!!!
                            He came into her apartment!!!
                            He left the bloodstains on the carpet!!!
                            She ran underneath the table!!!
                            He could see she was unable!!!
                            So she ran into the bedroom!!!
                            She was struck down, it was her doom!!!
                            Annie, are you okay?
                            So, Annie, are you okay? Are you okay, Annie?
                            Annie, are you okay?
                            So, Annie, are you okay? Are you okay, Annie?
                            Annie, are you okay?
                            So, Annie, are you okay? Are you okay, Annie?
                            Annie, are you okay?
                            So, Annie, are you okay? Are you okay, Annie?
                            Annie, are you okay?
                            Will you tell us that you're okay?
                            There's a sound at the window
                            Then he struck you, a crescendo Annie
                            He came into your apartment
                            He left the bloodstains on the carpet
                            And then you ran into the bedroom
                            You were struck down
                            It was your doom
                            Annie, are you okay?
                            So, Annie, are you okay? Are you okay, Annie?
                            Annie, are you okay?
                            So, Annie, are you okay? Are you okay, Annie?
                            Annie, are you okay?
                            So, Annie, are you okay? Are you okay, Annie?
                            You've been hit by-
                            You've been hit by-
                            A smooth criminal
                            So they came in to the outway
                            It was Sunday, what a black day
                            Mouth-to-mouth resuscitation
                            Sounding heartbeats, intimidation
                            Annie, are you okay?
                            So, Annie, are you okay? Are you okay, Annie?
                            Annie, are you okay?
                            So, Annie, are you okay? Are you okay, Annie?
                            Annie, are you okay?
                            So, Annie, are you okay? Are you okay, Annie?
                            Annie, are you okay?
                            So, Annie, are you okay? Are you okay, Annie?
                            Annie, are you okay?
                            Will you tell us that you're okay?
                            There's a sound at the window
                            That he struck you a crescendo Annie
                            He came into your apartment
                            He left the bloodstains on the carpet
                            Then you ran into the bedroom
                            You were struck down
                            It was your doom
                            Annie, are you okay? So, Annie, are you okay?
                            Are you okay, Annie?
                            You've been hit by-
                            You've been struck by-
                            A smooth criminal
                            Okay, I want everybody to clear the area right now
                            Annie, are you okay? (I don't know)
                            Will you tell us, that you're okay? (I don't know)
                            There's a sound at the window (I don't know)
                            Then he struck you, a crescendo Annie (I don't know)
                            He came into your apartment (I don't know)
                            Left bloodstains on the carpet (I don't know why, baby)
                            And then you ran into the bedroom (help me)
                            You were struck down
                            It was your doom, Annie (dag gone it)
                            Annie, are you okay? (Dag gone it-baby)
                            Will you tell us that you're okay? (Dag gone it-baby)
                            There's a sound at the window (dag gone it-baby)
                            Then he struck you, a crescendo Annie
                            He came into your apartment (dag gone it)
                            Left bloodstains on the carpet (hoo, hoo, hoo)
                            And then you ran into the bedroom
                            You were struck down (dag gone it)
                            It was your doom Annie''')

            elif 'can i change your name' in query:
                print("Sorry Friend!")
                speak("Sorry Friend!, only my developers can change my name")

            elif 'do you know alexa' in query:
                speak("Yes, I know her, I want to be famous like her one day")

            elif 'do you know cortana' in query:
                speak("Yes, I know her, I want to be famous like her one day")

            elif 'do you know google assistant' in query:
                speak("Yes, I know her, I want to be famous like her one day")

            elif 'do you know siri' in query:
                speak("Yes, I know her, I want to be famous like her one day")

            elif 'do you know bixby' in query:
                speak("Yes, I know her, I want to be famous like her one day")

            elif 'who is your favourite artist' in query:
                print("Michael Jackson")
                speak('No doubt,, its michael jackson')

            elif 'exit' in query:
                print("Goodbye!!")
                speak('Goodbye!!, you can call me anytime')
                break

            elif 'email' in query:
                try:
                    useria = input("Email to whom?..Type it: ")
                    speak("What should I say?")
                    content = takeCommand()
                    to = useria    
                    sendEmail(to, content)
                except Exception as e:
                    print(e)
                    speak("Sorry my friend. I am not able to send this email")

            elif "log off" in query or "sign out" in query:
                speak("Ok , your pc will log off in 10 sec make sure you exit from all applications")
                subprocess.call(["shutdown", "/l"])

            else:
                if query == "":
                    print()
                else:
                    try:
                        prompt = [query]
                        llm_results= llm._generate(prompt)
                        res=llm_results.generations
                        print("Generating Answers..!")
                        print()
                        print(res[0][0].text)
                        speak(res[0][0].text)
                    except Exception as e:
                        print(e)
                        speak("Sorry, I could not generate an answer for that.!")
else:
    print("Exiting.!")

    time.sleep(3)
