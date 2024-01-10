import os
import threading 
import time
# -- speech recognition 
import speech_recognition as sr
# -- chat gpt
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# -- text to speech
from gtts import gTTS

waiting_for_chat_gpt = False
chat_gpt_response = ''
chat_history = []


def play_text(text, use_audio=True):
    print("answer: " + text)
    if use_audio:
        tts = gTTS(text, lang='en')
        tts.save('response.mp3')
        os.system("vlc response.mp3 --play-and-exit")


def get_chat_gpt_response_threaded():
    global chat_history
    chat_history.append({"role": "user", "content": input_text})
    response = client.chat.completions.create(
        model="gpt-3.5-turbo", 
        messages=chat_history
    )
    global chat_gpt_response
    chat_gpt_response = response.choices[0].message.content
    global waiting_for_chat_gpt
    waiting_for_chat_gpt = False
    print("chat gpt finished: " + chat_gpt_response)

def say_greeting():
    # obtain audio from the microphone
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        threading.Thread(target=play_text('Hello! How can I assist you today?', use_audio=use_audio)).start()

use_audio = True

r = sr.Recognizer()
while(True):
    with sr.Microphone() as source:
        print("Listening for wake-up word...")
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)  # Adjust timeout as needed
    try:
        wake_up_word = r.recognize_google(audio).lower()
        if "hey" in wake_up_word:
            say_greeting()
            break
        else:
            print("Did not recognize wake-up word.")
    except sr.UnknownValueError:
        print("Could not understand the wake-up word.")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))

while(True):
    # obtain audio from the microphone
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)
        print(audio)
        print('listening finished')

    #convert audio to text
    try:
        # for testing purposes, we're just using the default API key
        # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
        # instead of `r.recognize_google(audio)`
        input_text = r.recognize_google(audio)

        waiting_for_chat_gpt = True
        threading.Thread(target=get_chat_gpt_response_threaded).start()       
        play_text("I understood: " + input_text)

        print("chat gpt time")
        print(waiting_for_chat_gpt)
        while waiting_for_chat_gpt:
            time.sleep(1)
            play_text("waiting for chat gpt... this can take some time...")

        play_text(chat_gpt_response)


    except sr.UnknownValueError:
        threading.Thread(target=play_text("Sorry, I could not understand", use_audio=True)).start()
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))