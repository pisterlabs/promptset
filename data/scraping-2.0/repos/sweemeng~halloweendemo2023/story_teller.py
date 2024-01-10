import os

import speech_recognition as sr
import pyttsx3
from dotenv import load_dotenv
load_dotenv()
import openai

MODEL = "gpt-3.5-turbo"


def main():
    recognizer = sr.Recognizer()
    speaker = pyttsx3.init()
    speaker.setProperty('rate', 200)
    system_prompt = "You're a scary assistant, you tell scary story to scare people for entertainment. "

    speaker.say("Do you want to hear a scary story?")
    speaker.runAndWait()
    with sr.Microphone() as source:
        while True:
            messages = [
                {"role": "system", "content": system_prompt},
            ]
            recognizer.adjust_for_ambient_noise(source)
            speaker.say("Tell me what kind of story you want to hear?")
            speaker.runAndWait()
            audio = recognizer.listen(source)

            text = recognizer.recognize_whisper_api(audio, api_key=os.getenv("OPENAI_API_KEY"))
            print("You said: {}".format(text))
            messages.append({"role": "user", "content": text})
            response = openai.ChatCompletion.create(
                model=MODEL,
                messages=messages,
                temperature=0
            )
            print(response)
            speaker.say(response["choices"][0]["message"]["content"])


if __name__ == '__main__':
    main()
