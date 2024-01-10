import os

import openai
import speech_recognition as sr
from gtts import gTTS
from playsound import playsound
from termcolor import colored

LANGUAGE = "es"  # define audio language
ENGINE_IA = "text-davinci-003"
AUDIO_FILE = "response.mp3"
openai.api_key = ""  # you api-key here


def voice():
    r = sr.Recognizer()

    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, duration=1)
        print("Ask me a question, speak clearly and slowly please => ")
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio, language="es-AR")
            print(text)
            new_question(LANGUAGE, ENGINE_IA, AUDIO_FILE, text)
        except:
            print("sorry, could not recognise")


def new_question(LANGUAGE, ENGINE_IA, AUDIO_FILE, text):
    try:
        print("Wait Moment....")
        completion = openai.Completion.create(
            engine=ENGINE_IA, prompt=text, n=1, max_tokens=2048)

        response_text = completion.choices[0].text
        print(response_text)

        tts = gTTS(response_text, lang=LANGUAGE, slow=False)
        tts.save(AUDIO_FILE)
        playsound(AUDIO_FILE)
        os.remove(AUDIO_FILE)
        print()
        print("Press Enter to continue...")
        input()  # Espera hasta que se presione Enter

    except Exception as e:
        print(colored("Sorry, error:", "red"))
        print(colored(str(e), "red"))

    print()


if __name__ == "__main__":
    while True:
        voice()
