import win32com.client
import speech_recognition as sr
from langchain_script import *
import webbrowser
import sys
import threading

# Constants
CHROME_PATH = "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe"
WAKE_WORD = "veronica"

# Initialize text-to-speech
speaker = win32com.client.Dispatch("SAPI.SpVoice")
voices = speaker.GetVoices()
female_voice = voices.Item(1)
speaker.Voice = female_voice

# Initialize speech recognition
r = sr.Recognizer()

# Flags for controlling the speech recognition thread
exit_flag = False

def take_input():
    with sr.Microphone() as mic:
        audio = r.listen(mic)
        try:
            text = r.recognize_google(audio, language="en-in")
            return text.lower()
        except Exception as e:
            print(e)
            return "Didn't hear clearly"

def listen_for_wake_word():
    global exit_flag
    global r

    while not exit_flag:
        text = take_input()
        if WAKE_WORD in text:
            speaker.Speak("How can I help you")
            process_user_input()

def process_user_input():
    text = take_input()
    if "stop" in text:
        speaker.Speak("It was nice assisting you")
        sys.exit()
    elif text != "Didn't hear clearly":
        if "open" in text:
            open_website(text)
        else:
            process_text_input(text)
    else:
        speaker.Speak(text)

def open_website(text):
    res = visitSite(user_message=text)
    site = res[12:(len(res)-4)]
    speaker.Speak(f"Opening {site} sir..")
    webbrowser.get(f"{CHROME_PATH} %s").open_new(res)

def process_text_input(text):
    res = getComplition(user_message=text)
    print(res)
    speaker.Speak(res)

# Main program loop
if __name__ == "__main__":
    # Start the speech recognition thread
    speech_thread = threading.Thread(target=listen_for_wake_word)
    speech_thread.start()

    while True:
        # Your main logic can now be focused on other tasks

        # Add a way to stop the program (for example, press Ctrl+C)
        try:
            pass  # Add your main logic here
        except KeyboardInterrupt:
            print("Exiting program.")
            exit_flag = True
            break

    # Wait for the speech recognition thread to finish
    speech_thread.join()
