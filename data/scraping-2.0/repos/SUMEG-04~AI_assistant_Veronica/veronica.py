import win32com.client
import speech_recognition as sr
import webbrowser
import sys
#from openai_script import *
from langchain_script import *
from agents.gmail_agent.gmail_agent import email
speaker=win32com.client.Dispatch('SAPI.SpVoice')
voices = speaker.GetVoices()
female_voice = voices.Item(1)
speaker.Voice = female_voice

import sounddevice as sd  # replaces pyannote.audio (uses imp)
import vosk  # lightweight keyword spotting library
from threading import Thread
import time

exit_flag = False

# Define wake word
WAKE_WORD = "veronica"

# Initialize keyword spotter
# vosk_model = vosk.Model(r"C:\Users\sumeg\Desktop\mernproject\veronics\server\vosk-model-small-en-us-0.15")    # Download and use your preferred vosk model
# vosk_recognizer = vosk.KaldiRecognizer(vosk_model,16000)



# Speech recognition and response generation
def take_input():
    print("taking input")
    r = sr.Recognizer()
    with sr.Microphone() as mic:
        r.pause_threshold= 1
        audio = r.listen(mic)
        try:
            text = r.recognize_google(audio, language="en-in")
            text = text.lower()
            return text
        except Exception as e:
            print(e)
            return "Didn't hear clearly"

# Main program loop
if __name__ == "__main__":
    speaker = win32com.client.Dispatch("SAPI.SpVoice")
    voices = speaker.GetVoices()
    female_voice = voices.Item(1)
    speaker.Voice = female_voice

    s = intro()
    speaker.Speak(s)
    while True: 
        print("Listening...")
        text = take_input()
        print(text)

        if WAKE_WORD in text:
            speaker.Speak("How can i help you")
            text = take_input()
            print(text)
            if "terminate" in text:
               speaker.Speak("It was nice assisting you")
               sys.exit()

            elif "email" or "mail" in text:
                res=email(text)
                speaker.Speak(res)

            elif text != "Didn't hear clearly":
                # Process text input based on keywords
                if f"open" in text:
                    res=visitSite(user_message=text)
                    site=res[12:(len(res)-4)]
                    speaker.Speak(f"Opening {site} sir..")
                    webbrowser.get("C:/Program Files (x86)/Google/Chrome/Application/chrome.exe %s").open_new(res)
                else:
                    res=getComplition(user_message=text)
                    print(res)
                    speaker.Speak(res)
            else:
                speaker.Speak(text)

        


