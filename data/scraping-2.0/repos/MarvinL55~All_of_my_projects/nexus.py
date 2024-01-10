import subprocess
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
import pyaudio
from twilio.rest import Client
from clint.textui import progress
from ecapture import ecapture as ec
from bs4 import BeautifulSoup
import win32com.client as wincl
from urllib.request import urlopen
import pyautogui
import nltk
import psutil
from nltk.sentiment import SentimentIntensityAnalyzer
import threading
import tkinter as tk
import cv2
import dlib
import datetime
import subprocess
import openai

openai.api_key = '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'


pyttsx3.init('sapi5')
engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

nltk.download("vader_lexicon")

sia = SentimentIntensityAnalyzer()

# Load the pre-trained detector
detector = dlib.get_frontal_face_detector()


def speak(audio):
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 0.8)
    engine.say(audio)
    engine.runAndWait()


def analyze_sentiment(text):
    # Perform sentiment analysis
    sentiment_scores = sia.polarity_scores(text)

    # Interpret the sentiment score
    if sentiment_scores['compound'] >= 0.05:
        return 'Positive'
    elif sentiment_scores['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'
    text = "I love this product"
    sentiment = analyze_sentiment(text)
    print(sentiment)


def wishMe():
    speak("Hello")
    speak(assname)

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
        return query

    except Exception as e:
        print(e)
        print("Unable to Recognize your voice.")
        return "None"

def username():
    data = load_user_data()
    if 'name' in data:
        name = data['name']
        speak("Welcome back, " + name + "!")
        print("Welcome back, " + name + "!")
    else:
        speak("What should i call you")
        name = takeCommand()
        data['name'] = name
        save_user_data(data)
        speak("Welcome, " + name + "!")
        print("Welcome, " + name + "!")
    speak("How can i help you?")

def load_user_data():
    try:
        with open('C:\\Users\\marvi\\PycharmProjects\\pythonProject1\\user_data.json', 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        data = {}
    return data

def save_user_data(data):
    with open('C:\\Users\\marvi\\PycharmProjects\\pythonProject1\\user_data.json', 'w') as file:
        json.dump(data, file)

def cpu():
    usage = str(psutil.cpu_percent())
    speak('CPU is at ' + usage)

def cpu_fan():
    speak("Name the process")
    process_name = takeCommand()
    process_name = process_name.lower()

    original_priotity = None

    for proc in psutil.process_iter():
        try:
            if process_name == proc.name().lower():
                process = psutil.Process(proc.pid)
                original_priotity = process.nice() # Store priority
                process.nice(psutil.HIGH_PRIORITY_CLASS)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

def battery():
    battery = psutil.sensors_battery()
    speak("Battery is at")
    speak(battery.percent)


def calc(query):
    try:
        client = wolframalpha.Client("U4A7V8-RK47GHT9L9")
        res = client.query(query)
        answer = next(res.results).text
        speak("The answer is " + answer)
    except Exception:
        speak("Sorry, I cant find an answer for that question")


def sendEmail(to, content):
    server = smtplib.SMTP('smpt.gmail.com', 587)
    server.ehlo()
    server.starttls()
    server.login("your email id", "your email password")
    server.sendmail("your mail id", to, content)
    server.close()

    if "send a email" in query:
        try:
            speak("What should i say")
            content = takeCommand()
            speak("Whom should i send it to")
            to = takeCommand()
            sendEmail(to, content)
        except Exception as e:
            print(e)
            speak("I am not able to send this email")


def screenshot():
    parent_directory = r"C:\NexusAI"
    directory = os.path.join(parent_directory, "screenshots")
    os.makedirs(directory, exist_ok=True)
    img = pyautogui.screenshot()
    img.save(os.path.join(directory, "screenshot.png"))


def get_news():
    try:
        api_key = "9818049735a542b5b783aef100db3d97"
        url = f"https://newsapi.org/v2/top-headlines?country=us&category=general&q=new%20york&apiKey={api_key}"
        jsonObj = urlopen(url)
        data = json.load(jsonObj)
        i = 1

        print("=============== NEW YORK NEWS ===============")
        speak("Here are some of the top headlines")

        for item in data['articles'][:2]:
            print(str(i) + '.' + item['title'] + '\n')
            speak(item['title'])
            speak("Next news")
            i += 1
    except Exception as e:
        print(str(e))


def tell_me_about(term):
    try:
        speak("Searching Wikipedia...")
        term = term.replace("wikipedia", "")
        results = wikipedia.summary(term, sentences=3)
        speak("According to wikipedia")
        print(results)
        speak(results)
    except Exception as e:
        speak("Sorry, I couldn't find any information about " + term)


def reminder():
    remindtime = takeCommand()
    speak("What should I remind you about?")
    remindtext = takeCommand()
    remindfile = open("remindfile.txt", "a")
    remindfile.write(remindtime + " " + remindtext + "\n")
    remindfile.close()
    speak("Reminder saved")

def cpu_fan():
    speak("Name the process")
    process_name = takeCommand()
    process_name = process_name.lower()

    for proc in psutil.process_iter():
        try:
            if process_name == proc.name().lower():
                process  = psutil.Process(proc.pid)
                process.nice(psutil.HIGH_PRIORITY_CLASS)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
             pass

def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = r.listen(source)
    try:
        query = r.recognize_google(audio)
        print(f"User said: {query}\n")
        return query
    except sr.UnknownValueError:
        print(e)
        print("Sorry I didn't catch that")
        return " "

def countdown_timer(x):
    while x >= 0:
        if x == 0:
            print("time is up")
            speak("time is up")
            break
        print(x)
        x -= 1
        time.sleep(1)

def google_search():
    speak("What should I search for?")
    search_query = takeCommand()
    url = f"https://www.google.com/search?q={search_query}"
    webbrowser.open(url)
    speak("Here are the results " + search_query)

def weather():
    api_key = "2107eb86cf98a0f11bc1e189774d9cd2"
    base_url = "https://api.openweathermap.org/data/2.5/weather?lat=40.659205&lon=-73.890690&appid=0023c10b6c1582cb4741a42787b3a2cd"
    city_name = "Brooklyn"
    complete_url = base_url + "appid = " + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    data = response.json()
    if data["cod"] != "404":
        main_data = data["main"]
        current_temperature = main_data["temp"]
        current_pressure = main_data["pressure"]
        current_humidity = main_data["humidity"]
        weather_data = data["weather"]
        weather_description = weather_data[0]["description"]
        print("Temperature (in Kelvin): " + str(current_temperature))
        print("Atmospheric pressure (in hPa): " + str(current_pressure))
        print("Humidity (in percentage): " + str(current_humidity))
        print("Description: " + str(weather_description))
    else:
        speak("City not found")


def face_detection():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            for face in faces:
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            username()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



def launch_app(application_path):
    os.startfile(application_path)


if __name__ == '__main__':
    clear = lambda: os.system('cls')

    clear()
    username()

    while True:

        query = takeCommand().lower()

        if "open Youtube" in query:
            speak("here is Youtube\n")
            webbrowser.open("youtube.com")

        elif "calculate" in query:
            speak("What is your calculation")
            calculation = takeCommand()
            calc(calculation)
            continue

        elif "battery percent" in query:
            speak(battery())

        elif "CPU" in query:
            speak(cpu())

        elif "fan" in query:
            speak(cpu_fan())

        elif "what is this song" in query:
            speak(lyrics())

        elif "what are the top stories" in query:
            speak(get_news())

        elif "open Google" in query:
            speak("her is Google happy searching\n")
            webbrowser.open("google.com")

        elif "open Amazon" in query:
            speak("opening amazon, happy shopping\n")
            webbrowser.open("amazon.com")

        elif "play music" in query:
            speak("here is music on apple music")
            webbrowser.open("applemusic.com")

        elif "what time is it" in query:
            strTime = datetime.datetime.now().strftime("% H: % M: % S")
            speak(f"The time is {strTime}")

        elif "set a reminder" in query:
            speak("What would you like to be reminded of")
            reminder()

        elif "tell me about " in query:
            term = query.replace("Tell me about", "")
            tell_me_about(term)

        elif "send a mail" in query:
            try:
                speak("What should I say?")
                content = takeCommand()
                speak("Whom should i send ")
                to = input()
                sendEmail(to, content)
                speak("Email has been sent")
            except Exception as e:
                print(e)
                speak("I am not able to send this email")

        elif "How are you" in query:
            speak("I'm fine thanks for asking")
            speak("How are you?")

        elif "fine" in query or "good" in query:
            speak("It's good to know that you are fine")

        elif "change my name to" in query:
            query = query.replace("change my name to", "")
            assname = query

        elif "change name" in query:
            speak("What would you like to call me")
            assname = takeCommand()
            speak("Thanks for naming me")

        elif "take a screenshot" in query:
            screenshot()
            speak("Screenshot taken")

        elif "what's your name" in query or "what is your name" in query:
            speak("My friends call me")
            speak(assname)
            print("My friends call me ", assname)

        elif "who made you" in query or "who created you" in query:
            speak("I have been created by Marvin")

        elif "exit" in query:
            speak("Thank you for the chat")

        elif "tell me a joke" in query:
            speak(pyjokes.get_joke())

        elif "who am i " in query:
            speak("If you talk then you are definitely a human")

        elif "why you came to the world" in query:
            speak("Thanks to Marvin. further its a secret")

        elif "power point presentation" in query:
            speak("opening google slides")
            webbrowser.open("googleslides.com")

        elif "what is love" in query:
            speak("It's the 7th sense that destroys all other sense, its very powerful")

        elif "who are you" in query:
            speak("I am your virtual assistant created by Marvin")

        elif "shutdown system" in query:
            speak("Hold on one second, Your system is shutting down")
            subprocess.call("shutdown / p /f")

        elif "empty recycling bin" in query:
            winshell.recycle_bin().empty(confirm=False, show_progress=False, sound=False)
            speak("Recycling bin emptied")

        elif "don't listen" in query or "stop listening" in query:
            speak("How long do you want me to stop listening for")
            a = int(takeCommand())
            time.sleep(a)
            print(a)

        elif "where is" in query:
            query = query.replace("where is ", "")
            location = query
            speak("User to Locate")
            speak(location)
            webbrowser.open("https://www.google.com/maps/place/ " + location + "")

        elif "open camera" in query or "take a picture" in query:
            ec.capture(0, "Jarvis camera", "img.jpg")

        elif "restart" in query:
            subprocess(["shutdown", "/r"])

        elif "sleep" in query or "hibernate" in query:
            speak("Laptop going to sleep")
            subprocess.call("shutdown /h")

        elif "log off" in query or "sign out" in query:
            speak("Make sure all the application are closed before signing out")
            time.sleep(5)
            subprocess.call(["shutdown", "/1"])

        elif "show notes" in query:
            speak("Showing Notes")
            file = open("jarvis.txt", "r")
            print(file.read())
            speak(file.read(6))

        elif "Ava" in query:

            wishMe()
            speak("Ava at your service")
            speak(assname)

        elif "what's the weather" in query:
            speak(weather())

        elif "open wikipedia" in query:
            webbrowser.open("wikipedia.com")

        elif "good morning " in query:
            speak("A warm " + query)
            speak("How are you")
            speak(assname)

        elif "how are you" in query:
            speak("I'm fine")

        elif "hello" in query:
            speak("Hi there")

        elif 'exit' in query:
            speak("Goodbye")
            exit()