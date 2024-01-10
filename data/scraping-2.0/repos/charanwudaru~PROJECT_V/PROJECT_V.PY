import os
import subprocess
import psutil
import pyttsx3
import requests
import json
import datetime
import calendar
import speech_recognition as sr
import openai
import face_recognition
import cv2
import webbrowser

# Initialize Text-to-Speech engine
try:
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)
except Exception as e:
    print(f"Error initializing text-to-speech: {str(e)}")
    engine = None

# Initialize Speech Recognition
recognizer = sr.Recognizer()
microphone = sr.Microphone()

def speak(text):
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"Error speaking text: {str(e)}")

def recognize_speech():
    try:
        with microphone as source:
            print("Listening...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)

        user_input = recognizer.recognize_google(audio).lower()
        print("You:", user_input)
        return user_input
    except sr.UnknownValueError:
        return ""
    except Exception as e:
        print(f"Error recognizing speech: {str(e)}")
        return ""

def check_cpu_usage():
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        print(f"Current CPU usage is {cpu_percent} percent.")
        speak(f"Current CPU usage is {cpu_percent} percent.")
    except Exception as e:
        print(f"Error checking CPU usage: {str(e)}")

def check_battery_usage():
    try:
        battery = psutil.sensors_battery()
        if battery:
            percent = battery.percent
            if battery.power_plugged:
                print(
                    f"The system is plugged in with {percent} percent battery remaining.")
                speak(
                    f"The system is plugged in with {percent} percent battery remaining.")
            else:
                print(
                    f"The system is running on battery with {percent} percent battery remaining.")
                speak(
                    f"The system is running on battery with {percent} percent battery remaining.")
        else:
            print("Unable to retrieve battery information.")
            speak("Unable to retrieve battery information.")
    except Exception as e:
        print(f"Error checking battery usage: {str(e)}")

def shutdown():
    try:
        speak("Shutting down the system.")
        os.system("shutdown /s /t 1")
    except Exception as e:
        print(f"Error shutting down the system: {str(e)}")

def restart():
    try:
        speak("Restarting the system.")
        os.system("shutdown /r /t 1")
    except Exception as e:
        print(f"Error restarting the system: {str(e)}")

def get_weather(city_name):
    try:
        api_key = "aa205ac2c9a4a3d031709f69d4742b11"
        base_url = "http://api.openweathermap.org/data/2.5/weather"
        params = {'q': city_name, 'appid': api_key, 'units': 'metric'}

        response = requests.get(base_url, params=params)
        data = response.json()

        if data.get('cod') == 200:
            temperature = data['main']['temp']
            weather_desc = data['weather'][0]['description']
            print(
                f"The weather in {city_name} is {weather_desc} with a temperature of {temperature} degrees Celsius.")
            speak(
                f"The weather in {city_name} is {weather_desc} with a temperature of {temperature} degrees Celsius.")
        else:
            speak("Unable to fetch weather information.")
    except Exception as e:
        print(f"Error getting weather information: {str(e)}")

def get_news():
    try:
        news_api_key = "1d9c4329385e4f6e94cb5931aefc941d"
        news_url = f"https://newsapi.org/v2/top-headlines?apiKey={news_api_key}&country=US"

        response = requests.get(news_url)
        data = response.json()

        if data['status'] == 'ok':
            articles = data['articles']
            speak("Here are the top news headlines.")
            for index, article in enumerate(articles[:3], start=1):
                title = article['title']
                print(f"Headline {index}: {title}")
                speak(f"Headline {index}: {title}")
        else:
            speak("Unable to fetch news updates.")
    except Exception as e:
        print(f"Error getting news updates: {str(e)}")

def get_calendar_events():
    try:
        today = datetime.date.today()
        day = today.day
        month = today.month
        year = today.year
        current_day = calendar.day_name[today.weekday()]
        print(
            f"Today is {current_day}, {day} {calendar.month_name[month]} {year}.")
        speak(
            f"Today is {current_day}, {day} {calendar.month_name[month]} {year}.")
    except Exception as e:
        print(f"Error getting calendar events: {str(e)}")

def wishme():
    try:
        speak("Welcome Back ")
        hour = datetime.datetime.now().hour
        if (hour >= 6 and hour < 12):
            speak("Good Morning sir!")
        elif (hour >= 12 and hour < 18):
            speak("Good afternoon sir")
        elif (hour >= 18 and hour < 24):
            speak("Good Evening sir")
        else:
            speak("Goodnight sir")
    except Exception as e:
        print(f"Error wishing user: {str(e)}")

def time():
    try:
        Time = datetime.datetime.now().strftime("%H")
        timetwo = datetime.datetime.now().strftime("%M")
        b = int(Time)
        if b >= 13:
            print(b - 12, end=':')
            print(timetwo, end=' ')
            print("pm")
            speak(b - 12)
            speak(timetwo)
            speak("pm")
        else:
            print(Time + timetwo + "am")
            speak(Time)
            speak(timetwo)
    except Exception as e:
        print(f"Error getting time: {str(e)}")

def relax():
    try:
        while (True):
            user_input = recognize_speech()
            if 'come on' in user_input or 'wake up' in user_input or 'jarvis' in user_input or 'hai' in user_input or 'hey jarvis' in user_input:
                speak('ya am ready for you')
                virtual_assistant()
            else:
                relax()
    except Exception as e:
        print(f"Error in relax mode: {str(e)}")

def virtual_assistant():
    try:
        while True:
            user_input = recognize_speech()

            if "cpu" in user_input:
                check_cpu_usage()
            elif "battery" in user_input:
                check_battery_usage()
            elif "shutdown" in user_input:
                shutdown()
            elif "restart" in user_input:
                restart()
            elif "weather" in user_input:
                city_name = 'Visakhapatnam'
                get_weather(city_name)
            elif "news" in user_input:
                get_news()
            elif "calendar" in user_input:
                get_calendar_events()
            elif('love you' in user_input or 'good' in user_input or 'very good' in user_input or 'your are smart' in user_input):
                speak("thank you")
                speak("its all because of you ")
                speak('love you')
            elif("hi" in user_input or 'hello' in user_input or 'hey' in user_input or 'hey' in user_input):
                speak("hi team i am ready to use")
            elif("off" in user_input or 'bye' in user_input or 'relax' in user_input or 'go off' in user_input):
                speak("bye bye... call me  if you need")
                speak('take care')
                print("wake me up if needed")
                relax()
            elif("time" in user_input):
                time()
            elif("online" in user_input or 'connect to ai' in user_input or 'gpt' in user_input):
                webbrowser.open("C:\\Users\\91630\\Desktop\\Project-V\\palamai.py")
                quit()
            elif 'open google' in user_input:
                speak('opening google')
                webbrowser.open_new("https://www.google.com")
            elif 'open youtube' in user_input:
                speak('opening youtube')
                webbrowser.open_new("https://www.youtube.com")
            elif 'open research paper' in user_input:
                speak('opening..')
                webbrowser.open_new("C:\\Users\\91630\\Desktop\\Project-V\\Research_Paper[1].docx")
    except Exception as e:
        print(f"Error in virtual assistant mode: {str(e)}")
        virtual_assistant()

if __name__ == "__main__":
        wishme()
        virtual_assistant()
    