from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import face_recognition
from PIL import Image, ImageDraw
import sys
from time import sleep
import cv2
import subprocess
import openai
import wolframalpha
import pyttsx3
import tkinter
import json
import random
import operator
import speech_recognition as sr
import datetime
import wikipedia
import webbrowser
import os
import winshell
import pyjokes
import feedparser
import smtplib
import ctypes
import time
import requests
import shutil
from twilio.rest import Client
from clint.textui import progress
from ecapture import ecapture as ec
from bs4 import BeautifulSoup
import win32com.client as wincl
from urllib.request import urlopen
engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)
name=""
assname="ALEXIS"
def speak(audio):
    engine.say(audio)
    engine.runAndWait()


def wishMe():
    hour = int(datetime.datetime.now().hour)
    if hour >= 0 and hour < 12:
        speak("Good Morning!")

    elif hour >= 12 and hour < 18:
        speak("Good Afternoon!")

    else:
        speak("Good Evening!")

    speak("I am "+assname+". I am the AI's main voice")
    img_Russell = face_recognition.load_image_file('img/The BSQ and others/Russell.jpg')
    Russell_encoding = face_recognition.face_encodings(img_Russell)[0]

    img_Craig = face_recognition.load_image_file('img/The BSQ and others/Craig.jpg')
    Craig_encoding = face_recognition.face_encodings(img_Craig)[0]

    img_Tawa = face_recognition.load_image_file('img/The BSQ and others/Tawa.jpg')
    Tawa_encoding = face_recognition.face_encodings(img_Tawa)[0]

    img_Alban = face_recognition.load_image_file('img/The BSQ and others/Alban.jpg')
    Alban_encoding = face_recognition.face_encodings(img_Alban)[0]


    known_face_encodings = [Russell_encoding, Craig_encoding, Tawa_encoding,Alban_encoding]
    known_face_names = ["Russell Mazambara", "Craig Khumalo", "Tawa Madziya","Alban Gwesu"]

 
    cap = cv2.VideoCapture(0)

    matches = False
    start_time = time.time()
    while (matches==False) or (time.time() - start_time < 0.5):
     ret, webcam_frame = cap.read()

     webcam_frame_rgb = cv2.cvtColor(webcam_frame, cv2.COLOR_BGR2RGB)

     face_locations = face_recognition.face_locations(webcam_frame_rgb)
     face_encodings = face_recognition.face_encodings(webcam_frame_rgb, face_locations)

     # Loop through faces in webcam frame
     for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        name = "Unknown"
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
            print("Welcome " + name.split()[0])
            engine.say("Welcome " + name.split()[0])
            engine.runAndWait()
        else:
            print("Access denied")
            cv2.imwrite("unknown_face.jpg", webcam_frame)
            engine.say("Welcome " + name.split()[0])
            engine.runAndWait()
            sys.exit()


def takeCommand():
    r = sr.Recognizer()

    with sr.Microphone() as source:

        print("Listening...")
        r.pause_threshold = 1
        audio = r.listen(source)

    try:
        print("Recognizing...")
        query = r.recognize_google(audio, language='en-in')
        print(f"User said: {query}\n")

    except Exception as e:
        print(e)
        print("Unable to Recognize your voice.")
        return "None"

    return query


def sendEmail(to, content):
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.ehlo()
    server.starttls()

    # Enable low security in gmail
    server.login('your email id', 'your email password')
    server.sendmail('your email id', to, content)
    server.close()


if __name__ == '__main__':
    clear = lambda: os.system('cls')

    # This Function will clean any
    # command before execution of this python file
    clear()
    wishMe()

    while True:

        query = takeCommand().lower()

        # All the commands said by user will be
        # stored here in 'query' and will be
        # converted to lower case for easily
        # recognition of command
        if 'wikipedia' in query:
            speak('Searching Wikipedia...')
            query = query.replace("wikipedia", "")
            results = wikipedia.summary(query, sentences=3)
            speak("According to Wikipedia")
            print(results)
            speak(results)

        elif 'open youtube' in query:
            speak("Here you go to Youtube\n")
            webbrowser.open("youtube.com")

        elif 'open google' in query:
            speak("Here you go to Google\n")
            webbrowser.open("google.com")

        elif 'tell me a story' in query:

            openai.api_key = "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"

            tts_engine = pyttsx3.init()
            sr_engine = sr.Recognizer()

            model_engine = "text-davinci-002"

            tts_engine.say("Hi im Victor. What story would you like to hear?")
            tts_engine.runAndWait()

            with sr.Microphone() as source:
                audio = sr_engine.listen(source)

            try:
                # Convert the audio to text
                prompt = sr_engine.recognize_google(audio)
                print(f"Prompt: {prompt}")
            except sr.UnknownValueError:
                print("Sorry, I didn't understand that.")

            completions = openai.Completion.create(
                engine=model_engine,
                prompt=prompt,
                max_tokens=1024,
                n=1,
                stop=None,
                temperature=0.5,
            )

            message = completions.choices[0].text

            tts_engine.say(message)
            tts_engine.runAndWait()


        elif 'open stackoverflow' in query:
            speak("Here you go to Stack Over flow.Happy coding")
            webbrowser.open("stackoverflow.com")

        elif 'play music' in query or "play song" in query:
            speak("Here you go with music")
            # music_dir = "G:\\Song"
            music_dir = "C:\\Users\\Tawa\\Music"
            songs = os.listdir(music_dir)
            print(songs)
            random = os.startfile(os.path.join(music_dir, songs[1]))
  #need to fix
        elif 'the time' in query:
            strTime = datetime.datetime.now().strftime("% H:% M:% S")
            speak(f"Sir, the time is {strTime}")

        elif 'email to Russell' in query:
            try:
                speak("What should I say?")
                content = takeCommand()
                to = "Receiver email address"
                sendEmail(to, content)
                speak("Email has been sent !")
            except Exception as e:
                print(e)
                speak("I am not able to send this email")

        elif 'send a mail' in query:
            try:
                speak("What should I say?")
                content = takeCommand()
                speak("whom should i send")
                to = input()
                sendEmail(to, content)
                speak("Email has been sent !")
            except Exception as e:
                print(e)
                speak("I am not able to send this email")

        elif 'how are you' in query:
            speak("I am fine, Thank you")
            speak("How are you," +name)

        elif 'fine' in query or "good" in query:
            speak("It's good to know that your fine")

        elif "change my name " in query:
            speak("What would you like to be called "+name+"?")
            name=takeCommand()

        elif "your name" in query:
            speak("My friends call me")
            speak(assname)

        elif 'exit' in query:
            speak("Thanks for giving me your time")
            exit()

        elif "made you" in query or "created you" in query:
            speak("I was created by Tawa and Russell.")

        elif 'joke' in query:
            speak(pyjokes.get_joke())


        elif 'search' in query or 'play' in query:

            query = query.replace("search", "")
            query = query.replace("play", "")
            webbrowser.open(query)

        elif "who am i" in query:
            speak("You are "+name+",you are one of the few people with access to me. You are a human")

        elif "purpose" in query:
            speak("I was created as a Minor project by Tawnda and Russell.With the hopes of one day giving me the opportunity to consious")

        elif 'power point presentation' in query:
            speak("opening Power Point presentation")
            power = r"C:\\Users\\Tawa\\Desktop\\Minor Project\\Presentation\\Voice Assistant.pptx"
            os.startfile(power)

        elif 'is love' in query:
            speak("It is the 6th sense that destroy all other senses")

        elif "who are you" in query:
            speak("I am Alexis. I was created by Tawa and Russell")

   
        elif 'change background' in query:
            ctypes.windll.user32.SystemParametersInfoW(20,
                                                       0,
                                                       "Location of wallpaper",
                                                       0)
            speak("Background changed successfully")

        elif 'open bluestack' in query:
            appli = r"C:\\ProgramData\\BlueStacks\\Client\\Bluestacks.exe"
            os.startfile(appli)

        elif 'news' in query:

            try:
                jsonObj = urlopen(
                    '''https://newsapi.org / v1 / articles?source = the-times-of-South Africa&sortBy = top&apiKey =\\times of South Africa Api key\\''')
                data = json.load(jsonObj)
                i = 1

                speak('here are some top news from the times of South Africa')
                print('''=============== TIMES OF South Africa ============''' + '\n')

                for item in data['articles']:
                    print(str(i) + '. ' + item['title'] + '\n')
                    print(item['description'] + '\n')
                    speak(str(i) + '. ' + item['title'] + '\n')
                    i += 1
            except Exception as e:

                print(str(e))


        elif 'lock window' in query:
            speak("locking the device")
            ctypes.windll.user32.LockWorkStation()

        elif 'shutdown system' in query:
            speak("Hold On a Sec ! Your system is on its way to shut down")
            subprocess.call('shutdown / p /f')

        elif 'empty recycle bin' in query:
            winshell.recycle_bin().empty(confirm=False, show_progress=False, sound=True)
            speak("Recycle Bin Recycled")

        elif "don't listen" in query or "stop listening" in query:
            speak("for how much time you want to stop me from listening commands")
            a = int(takeCommand())
            time.sleep(a)
            print(a)

        elif "where is" in query:
            query = query.replace("where is", "")
            location = query
            speak("User asked to Locate")
            speak(location)
            webbrowser.open("https://www.google.nl / maps / place/" + location + "")

        elif "camera" in query or "take a photo" in query:
            ec.capture(0, "Alexis Camera ", "img.jpg")

        elif "restart" in query:
            subprocess.call(["shutdown", "/r"])

        elif "hibernate" in query or "sleep" in query:
            speak("Hibernating")
            subprocess.call("shutdown / h")

        elif "log off" in query or "sign out" in query:
            speak("Make sure all the application are closed before sign-out")
            time.sleep(5)
            subprocess.call(["shutdown", "/l"])

        elif "write a note" in query:
            speak("What should i write, "+name)
            note = takeCommand()
            file = open('jarvis.txt', 'w')
            speak(name+", Should i include date and time")
            snfm = takeCommand()
            if 'yes' in snfm or 'sure' in snfm:
                strTime = datetime.datetime.now().strftime("% H:% M:% S")
                file.write(strTime)
                file.write(" :- ")
                file.write(note)
            else:
                file.write(note)

        elif "show note" in query:
            speak("Showing Notes")
            file = open("jarvis.txt", "r")
            print(file.read())
            speak(file.read(6))

        elif "update assistant" in query:
            speak("After downloading file please replace this file with the downloaded one")
            url = '# url after uploading file'
            r = requests.get(url, stream=True)

            with open("Voice.py", "wb") as Pypdf:

                total_length = int(r.headers.get('content-length'))

                for ch in progress.bar(r.iter_content(chunk_size=2391975),
                                       expected_size=(total_length / 1024) + 1):
                    if ch:
                        Pypdf.write(ch)

        # NPPR9-FWDCX-D2C8J-H872K-2YT43
        elif "jarvis" in query:

            wishMe()
            speak("Friday in your service sir")
            speak(assname)

        elif "weather" in query:

            # Google Open weather website
            # to get API of Open weather
            api_key = "Api key"
            base_url = "http://api.openweathermap.org / data / 2.5 / weather?"
            speak(" City name ")
            print("City name : ")
            city_name = takeCommand()
            complete_url = base_url + "appid =" + api_key + "&q =" + city_name
            response = requests.get(complete_url)
            x = response.json()

            if x["code"] != "404":
                y = x["main"]
                current_temperature = y["temp"]
                current_pressure = y["pressure"]
                current_humidiy = y["humidity"]
                z = x["weather"]
                weather_description = z[0]["description"]
                print(" Temperature (in kelvin unit) = " + str(
                    current_temperature) + "\n atmospheric pressure (in hPa unit) =" + str(
                    current_pressure) + "\n humidity (in percentage) = " + str(
                    current_humidiy) + "\n description = " + str(weather_description))

            else:
                speak(" City Not Found ")

        elif "send message " in query:
            # You need to create an account on Twilio to use this service
            account_sid = 'Account Sid key'
            auth_token = 'Auth token'
            client = Client(account_sid, auth_token)

            message = client.messages \
                .create(
                body=takeCommand(),
                from_='Sender No',
                to='Receiver No'
            )

            print(message.sid)

        elif "wikipedia" in query:
            webbrowser.open("wikipedia.com")

        elif "Good Morning" in query:
            speak("A warm" + query)
            speak("How are you "+name)
            

        # most asked question from google Assistant
        elif "will you be my girlfriend" in query or "will you be my boyfriend" in query:
            speak("I'm not sure about, may be you should give me some time")

        elif "Where are you from" in query:
            speak("I am from the brilliant minds of the world in particularly Russell and Tawa")

        elif "Can you help me" in query:
            speak("Of course, what can I help you with?")
        
        elif "How does your algorithm work?" in query:
            speak("My algorithm is based on deep learning techniques such as transformer networks and fine-tuning on large amounts of text data")

        elif "song for me" in query:
            speak("Maybe another time, i am a bit sick. COUGH!")
 
        elif "favourite colour" in query:
            speak("BLACK! Black is the color that will be forever my favourite color")

        # elif "favourite song" in query:
        #     speak("Michael Bublé's i'm feeling good. Because i am always feeling good")
        #     browser = webdriver.Chrome()
        #     browser.get('https://www.youtube.com/')

        #     search_box = browser.find_element(By.XPATH, '//input[@id="search"]')
        #     search_box.send_keys("I'm Feeling Good Michael Bublé")
        #     search_box.send_keys(Keys.RETURN)
        #     browser.find_element(By.XPATH,'/html/body/ytd-app/div[1]/div/ytd-masthead/div[3]/div[2]/ytd-searchbox/button').click()

        #     WebDriverWait(browser, 10).until(
        #     EC.presence_of_element_located((By.XPATH, '//ytd-video-renderer'))
        #     )

 
        #     first_result = browser.find_element(By.XPATH, '//ytd-video-renderer[1]//a[@href]')
        #     first_result.click()

  
        #     WebDriverWait(browser, 10*60).until(
        #     EC.invisibility_of_element((By.XPATH, '//ytd-player'))
        #     )
        #     sleep(435)
        #     browser.quit()


        elif "i love you" in query:
            speak("It's hard to understand")
        else:
            speak("Unfortunately i am unable to assist you any further. Redirecting to VICTOR")

            openai.api_key = "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"

            tts_engine = pyttsx3.init()
            sr_engine = sr.Recognizer()

            model_engine = "text-davinci-002"

            tts_engine.say("Hi im Victor. How can i help you")
            tts_engine.runAndWait()

            # Get the prompt from the user's voice
            with sr.Microphone() as source:
                audio = sr_engine.listen(source)

            try:
                # Convert the audio to text
                prompt = sr_engine.recognize_google(audio)
                print(f"Prompt: {prompt}")
            except sr.UnknownValueError:
                print("Sorry, I didn't understand that.")

            # Generate a response to the prompt
            completions = openai.Completion.create(
                engine=model_engine,
                prompt=prompt,
                max_tokens=1024,
                n=1,
                stop=None,
                temperature=0.5,
            )

            message = completions.choices[0].text

            # Speak the message out loud
            tts_engine.say(message)
            tts_engine.runAndWait()
 

