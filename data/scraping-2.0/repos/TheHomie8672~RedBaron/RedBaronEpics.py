#  E-P.I.C.S  or   (Experimental Paralell Intermitent Command Sequence)

# E.P.I.C.S is intended to be a replacement for the current Main_Loop() Sequence. E.P.I.C.S will be

import concurrent.futures
import pyttsx3
import pyaudio
import wave
import speech_recognition as sr
import time
import openai
import webbrowser
import validators
import GPTPrimaryAgent 
from GPTPrimaryAgent import PrimaryAgentResponse, say_output, get_input


# Set OpenAI API key
openai.api_key = "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"

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Define function to speak text
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Define parameters for the text generation request
model = "text-davinci-003"
temperature = 0.3
max_tokens = 2048

# Define main loop
def main_loop():
    r = sr.Recognizer()
    keyword = "hey baron"
    last_time = time.time()
    prompt = "your name is Baron, you are an assistant created by Marcus Sherman to assist with coding problems and general help. try to avoid long-winded answers. you are currently talking to Marcus Sherman."
    while True:
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source, duration=1)
            audio = r.listen(source)
        try:
            text = r.recognize_google(audio).lower()
            if keyword in text:
                # Prompt user for input
                speak("What can I help you with?")
                with sr.Microphone() as source:
                    r.adjust_for_ambient_noise(source, duration=1)
                    audio = r.listen(source, timeout=5)
                try:
                    user_input = r.recognize_google(audio)
                    # Pass input to GPT-3 API
                    response = openai.Completion.create(
                        engine=model,
                        prompt=prompt + user_input,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    # Speak response
                    speak(response.choices[0].text)
                    last_time = time.time()
                except sr.WaitTimeoutError:
                    # No input detected after 5 seconds, ask for keyword again
                    print("say 'Hey Baron' to activate me!  ")
                    last_time = time.time()
                except sr.UnknownValueError:
                    pass
        except sr.UnknownValueError:
            pass
        # Wait for 5 seconds before asking for keyword again
        if time.time() - last_time > 5:
            print("say 'Hey Baron' to activate me!")
            last_time = time.time()

# Start main loop
if __name__ == "__main__":
    main_loop()
