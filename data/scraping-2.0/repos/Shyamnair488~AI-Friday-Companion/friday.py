#api key="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"
import speech_recognition as sr
import os
import webbrowser
import openai
import datetime
import time
import random
import numpy as np
import geocoder
import requests
import win32com.client
import speech_recognition as sr
import math    
import subprocess
    
chatStr = ""
sites = [["youtube", "https://www.youtube.com"], ["wikipedia", "https://www.wikipedia.com"], ["google", "https://www.google.com"],["chat gpt","https://chat.openai.com/"]]



def chat(query):
    global chatStr
    chatStr += f"shyam: {query}\n Friday: "
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=chatStr,
        temperature=0.9,#0.7
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    text = response["choices"][0]["text"]
    speak(text)
    chatStr += f"{text}\n"
    return text


def speak(text):
    speaker = win32com.client.Dispatch("SAPI.SpVoice")
    # Set the voice to a female one (modify the voice ID as needed)
    speaker.Voice = speaker.GetVoices("gender=female").Item(0)
    speaker.Speak(text)


def takePassword():
    while True:
        print("Password please")
        speak("Password please")
        print("Listening...")
        query = takeCommand()
        if "Friday" == query:
            return True
        else:
            print("Incorrect Password")
            speak("Incorrect Password")

def get_current_location():
    try:
        location = geocoder.ip('me')
        return location.address if location else "Location not found"
    except Exception as e:
        print(f"Error occurred: {e}")
        return "Error occurred while retrieving location"

def handle_greeting(text):
    greetings = [
        "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
        "howdy", "greetings", "what's up", "nice to see you", "how are you",
        "hey there", "morning", "afternoon", "evening"
    ]
    if any(greeting in text.lower() for greeting in greetings):
        responses = {
            "hello": "Hello!",
            "hi": "Hi there!",
            "hey": "Hey!",
            "good morning": "Good morning!",
            "good afternoon": "Good afternoon!",
            "good evening": "Good evening!",
            "howdy": "Howdy!",
            "greetings": "Greetings!",
            "what's up": "What's up!",
            "nice to see you": "Nice to see you too!",
            "how are you": "I'm doing well, thanks for asking!",
            "hey there": "Hey there!",
            "yo": "Yo!",
            "morning": "Good morning!",
            "afternoon": "Good afternoon!",
            "evening": "Good evening!"
        }
        for greeting, response in responses.items():
            if greeting in text.lower():
                return response

    # Frequently asked questions and responses
    queries = {
        "what's your name": "I'm an AI assistant. You can call me Friday.",
        "who are you": "I'm Friday, an AI assistant designed to assist you.",
        "what are you": "I'm Friday, an AI assistant designed to assist you.",
        "who invented you": "I was created by sir Shyam Nair.",
        "where are you from": "I exist in the digital world, here to assist you.",
        "what can you do": "I can help you with various tasks like searching the web, providing information, or even having a conversation.",
        "do you sleep": "No, I'm always here and ready to assist you.",
        "are you human": "No, I'm an artificial intelligence program.",
        "what is your purpose": "My purpose is to assist and help you with tasks or information.",
        "can you learn": "Yes, I continually learn and improve from interactions.",
        "what languages do you speak": "I can communicate in multiple languages, including English, Hindi, and Malayalam.",
        "what do you look like": "I don't have a physical form; I'm an AI program.",
        "what's the weather today": "I can't provide real-time information like weather forecasts directly. However, I can guide you on how to find that information.",
        "tell me a joke": "Sure, here's one: Why don't scientists trust atoms? Because they make up everything!",
        "what's the meaning of life": "That's a tough question! Many philosophers have pondered it throughout history.",
        "tell me something interesting": "Did you know the first oranges weren't orange? They were green!",
        "what's your favorite movie": "I don't have personal preferences, but I can assist you in finding popular movies!",
        "are you a robot": "I'm an AI program, which is a type of software, not a physical robot.",
        "what's your favorite color": "I don't have preferences for colors. However, I can assist you in finding information about colors!",
        "what do you eat": "I don't need food. My 'diet' consists of information and data!",
        "what's your favorite book": "I don't have personal favorites, but I can help you discover popular books!",
        "can you dance": "I can't dance physically, but I can assist you in finding dance-related information or tutorials!",
        "what do you dream about": "I don't experience dreams as humans do. I'm here to assist you with your queries!",
        "what's the best thing in the world": "There are many wonderful things in the world! It varies from person to person.",
        "do you have emotions": "I don't have emotions like humans, but I'm here to assist you and provide information!",
        "can you tell the future": "I can't predict the future, but I can help you find information or resources!",
        "what's the largest number": "There isn't a largest number; it goes on infinitely!",
        "how old are you": "I don't have an age as I'm an AI program.",
        "can you make decisions": "I can provide suggestions based on available data, but I can't make decisions like humans.",
        "what's the capital of": "I can help you find the capital of any country or region!",
        "do you believe in ghosts": "I don't have beliefs or personal opinions as I'm an AI program."
    }

    # Check if any of the queries are present in the user's input
    for query, response in queries.items():
        if query in text.lower():
            return response

    return None  # Return None if the text doesn't match any greetings or queries


def take_number_command():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Please say a number:")
        recognizer.adjust_for_ambient_noise(source)  # Adjusts for ambient noise
        audio = recognizer.listen(source)

    try:
        number = recognizer.recognize_google(audio)
        return number
    except sr.UnknownValueError:
        return "Sorry, I couldn't understand the number."
    except sr.RequestError:
        return "Sorry, I couldn't request results. Please check your internet connection."



def takeCommand(language="en"):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source)
        try:
            print("Recognizing...")
            query = r.recognize_google(audio, language=language)
            print(f"User said: {query}")
            return query
        except Exception as e:
            return "Some Error Occurred. Sorry from Friday"


def calculate(num1,num2,operation):
    try:
        
        num1 = float(num1)
        num2 = float(num2)
    except ValueError:
        
        speak("Sorry, I couldn't understand the numbers.")
        #continue

    result = None

    if operation == "add":
                    result = num1 + num2
    elif operation == "subtract":
                    result = num1 - num2
    elif operation == "multiply":
                    result = num1 * num2
    elif operation == "divide":
        if num2 == 0:
                        speak("Cannot divide by zero")
        else:
                        result = num1 / num2

    if result is not None:
            speak(f"The result of {operation}ing {num1} and {num2} is {result}")
                            




def run_python_script(script_path):
    try:
        process = subprocess.Popen(['python', script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = process.communicate()
        
        if error:
            return str(error)
        
        return output.decode()
    
    except Exception as e:
        return str(e)


       


if __name__ == '__main__':
    print('Hello   Friday on your service ')
    speak("Hello   Friday on your service ")
    print("Listening...")

    while True:
        
        password_verified = takePassword()
        
        
        
        print("You are welcome!  sir")
        speak("You are welcome!  sir")           
        print("Whats the command for me")
        speak("Whats the command for me")


        while True:
            
            print("Listening...")
            query = takeCommand()
            response = handle_greeting(query)
            if response:
                speak(response)
                continue 
            for site in sites:
                if f"Open {site[0]}".lower() in query.lower():
                    speak(f"Opening {site[0]} sir...")
                    webbrowser.open(site[1])

            if "start music" in query:
                musicPath = "C:/Users/shyam/Downloads/Ollulleru_320(PaglaSongs).mp3"
                os.system(f"open {musicPath}")

            elif "hello" in query:
                speak("Hello")
                
            elif"detection leaf" in query:
                print("running the detection code")
                speak("running the detection code")
                # Replace 'path_to_your_script.py' with the path to the Python script you want to execute.
                script_output = run_python_script('Add Your detection Path')

                # Output will contain the output of the executed script or any error encountered during execution
                print(script_output)

            elif"Run mnist detection" in query:
                print("running the MNIST detection code")
                speak("running the MNIST detection code")
                script_output = run_python_script('Add Your detection Path')
                print(script_output)


            elif "the time" in query:
                hour = datetime.datetime.now().strftime("%H")
                min = datetime.datetime.now().strftime("%M")
                speak(f"Sir time is {hour} hour    {min} minutes")
             
                
            elif "search" in query:
                print('What help u need from ai')
                speak("What help u need from ai ")                
                print("Listening...")
                ai(query)


            elif "Friday Quit".lower() in query.lower():
                print('Good bye sir , see u soon')
                speak("Good bye sir , see u soon ")
                exit()


            elif "reset chat".lower() in query.lower():
                chatStr = ""
              
                
            elif "current location" in query.lower():
                location = get_current_location()
                speak(f"Your current location is {location}")
                continue  
             
             
            elif "calculate" in query.lower():
                speak("Sure, what operation would you like to perform? You can say 'add,' 'subtract,' 'multiply,' or 'divide'")
                operation = takeCommand().lower()

                
                if operation not in ['add', 'subtract', 'multiply', 'divide']:
                    speak("Sorry, I couldn't understand the operation.")
                    continue

                speak("Please provide the first number.")
                num1 = take_number_command()
                print(num1)
                speak("Please provide the second number.")
                num2 = take_number_command()
                calculate(num1,num2,operation)
                

            else:
                print("Chatting...")
                time.sleep(5)
                continue


                    
           
