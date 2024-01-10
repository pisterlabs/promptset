#!/usr/bin/env python
#
# Design Kei Sawamura
# Author Kei Sawamura
#
# This application allows you to call GPT with voice.
#

import openai
import os
import speech_recognition as sr

from gtts import gTTS
from playsound import playsound


def listen_to_order():

    # Create a recognizer instance.
    r = sr.Recognizer()
    
    # Continuously listen for order.
    with sr.Microphone() as source:
        print("Listening...")
        audio = r.listen(source)

    try:
        # Use Google's speech recognition service to convert speech to text.
        content = r.recognize_google(audio, language='ja-JP')
        print("You said: {}".format(content))

        return content

    except sr.UnknownValueError:
        # If the speech was unclear, it will throw this error.
        print("Sorry, I didn't catch that.")

    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))
    


def call_openai_api(content, model = "gpt-4"):

    # Load your API key from an environment variable or secret management service.
    openai.api_key = os.getenv("OPENAI_API_KEY")
    response = openai.ChatCompletion.create(model = model,
                messages = [
                        { "role": "user", "content": "次の日本語の質問を英語で回答してください" },
                        { "role": "assistant", "content": "OK, How can I help you?" },
                        { "role": "user", "content": content }
                    ]
                )

    return response



def save_and_speach(context):

    res = call_openai_api(content)

    msg = res.choices[0].message['content']
    mp3 = res.id + ".mp3"
    txt = res.id + ".txt"

    print(msg)

    os.mkdir(content)
    tts = gTTS(msg, lang='en')
    tts.save(content + os.sep + mp3)
    playsound(content + os.sep + mp3)

    with open(content + os.sep + txt, "w") as file:
        file.write(msg)



if __name__ == '__main__':

    content = listen_to_order()
    ans = input("Do you call GPT-API? [y/n]:  ")

    if (ans == "y"):
        save_and_speach(content)
    

