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

# Initialize Text-to-Speech engine
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)

# Initialize Speech Recognition
recognizer = sr.Recognizer()
microphone = sr.Microphone()


def speak(text):
    engine.say(text)
    engine.runAndWait()


def recognize_speech():
    with microphone as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        user_input = recognizer.recognize_google(audio).lower()
        print("You:", user_input)
        return user_input
    except sr.UnknownValueError:
        return ""


def check_cpu_usage():
    cpu_percent = psutil.cpu_percent(interval=1)
    print(f"Current CPU usage is {cpu_percent} percent.")
    speak(f"Current CPU usage is {cpu_percent} percent.")


def generate_response(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150
        )
        return response.choices[0].message["content"]

    except Exception as e:
        print(f"Error generating response: {str(e)}")
        return "I encountered an error."


def check_battery_usage():
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


def shutdown():
    speak("Shutting down the system.")
    os.system("shutdown /s /t 1")


def restart():
    speak("Restarting the system.")
    os.system("shutdown /r /t 1")


def get_weather(city_name):
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


def get_news():
    news_api_key = "1d9c4329385e4f6e94cb5931aefc941d"
    news_url = f"https://newsapi.org/v2/top-headlines?apiKey={news_api_key}&country=US"

    response = requests.get(news_url)
    data = response.json()

    if data['status'] == 'ok':
        articles = data['articles']
        speak("Here are the top news headlines.")
        for index, article in enumerate(articles[:5], start=1):
            title = article['title']
            print(f"Headline {index}: {title}")
            speak(f"Headline {index}: {title}")
    else:
        speak("Unable to fetch news updates.")


def get_calendar_events():
    today = datetime.date.today()
    day = today.day
    month = today.month
    year = today.year
    current_day = calendar.day_name[today.weekday()]
    print(
        'f"Today is {current_day}, {day} {calendar.month_name[month]} {year}."')
    speak(
        f"Today is {current_day}, {day} {calendar.month_name[month]} {year}.")


def generate_response(prompt):

    api_key = "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"

    openai.api_key = api_key
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150
        )
        return response.choices[0].message["content"]

    except Exception as e:
        print(f"Error generating response: {str(e)}")
        return "I encountered an error."


speak("Hello buddy!")


def virtual_assistant():
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
        elif "exit" in user_input or "quit" in user_input:
            speak("Goodbye!")
            break
        elif "ai" in user_input or "open ai" in user_input:
            speak("connecting to open AI chat GPT")
            while True:
                voice_input = recognize_speech()
                if "exit" in voice_input:
                    speak("closeing GPT")
                    virtual_assistant()
                elif voice_input:
                    response = generate_response(voice_input)
                    print("ChatGPT:")
                    print(response)
                    speak(response[1:100])
                    with open("response.txt", "a") as file:
                        file.write(response + "\n")


virtual_assistant()
