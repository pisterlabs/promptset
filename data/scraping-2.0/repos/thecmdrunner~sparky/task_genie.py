# Trying out chatGPT api with voice commands.

import speech_recognition as sr
import datetime
import time
import os
# from playsound import playsound
import requests
import json
import openai
import pyautogui
from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env file

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY
# openai.api_key = os.environ["OPENAI_API_KEY"]

doneListening = './audio/done_listening.wav'

ChatCompletionsURL = 'https://api.openai.com/v1/chat/completions'
OpenAI_Headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {OPENAI_API_KEY}'
}

# Fill your own details

yourName = 'Master'  # enter your name

# def beep():
#     playsound(doneListening)


def takeCommand():
    # take microphone input from the user and returns a string output
    print('Now say something')
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Listening for command...")
        recognizer.pause_threshold = 1.5
        audio = recognizer.listen(source)

    try:
        print("Recognizing...")
        audiowords = recognizer.recognize_google(audio, language='en-in')
        print("You said: ", audiowords, "\n")

    except Exception as e:
        print(e)
        print("Say that again please...")
        return "None"

    return audiowords


def respond(query):

    try:
        print("You said: " + query)

        if query == 'None':
            return 0

        elif ('bye' or 'exit' or 'shut down' or 'shutdown') in query:
            # beep()
            print("Turning off!")
            exit()

        # generate a response using OpenAI ChatGPT API
        prompt = f"What is your opinion on {query}?"
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role":
                "system",
                "content":
                "You are a programming assistant. Respond only in javascript code. Keep all other prose as javascript comments. Strictly output nothing else, other than javascript code. Do not include markdown code syntax."
            }, {
                "role": "user",
                "content": query
            }])
        response = completion.choices[0].message.content.strip()

        # data = {
        #     'model': 'gpt-3.5-turbo',
        #     'stream': True,
        #     'messages': [{
        #         'role': 'user',
        #         'content': query
        #     }]
        # }

        # r = requests.post(ChatCompletionsURL,
        #                            headers=OpenAI_Headers,
        #                            data=json.dumps(data))

        # response = r.content
        print(response)

        pyautogui.typewrite(response)

        # for char in response:
        # pyautogui.press(char)
        # time.sleep(0.1)  # add a small delay between each key press

        return

    except Exception as e:
        #
        print("Exception occured:")
        print(e)
        print('\nPlease try again...\n')


if __name__ == "__main__":
    # beep()
    while True:
        # beep()
        order = takeCommand()
        respond(order)
