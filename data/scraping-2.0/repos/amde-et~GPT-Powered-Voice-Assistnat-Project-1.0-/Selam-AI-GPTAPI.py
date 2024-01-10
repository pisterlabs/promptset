from os import system
import speech_recognition as sr
import openai  # Import the OpenAI library
import warnings
import time
import os

# Set your OpenAI API key
openai.api_key = "YOUR_OPENAI_API_KEY"

wake_word = 'Selam'
listening_for_wake_word = True
source = sr.Microphone()
warnings.filterwarnings("ignore", category=UserWarning, module='whisper.transcribe', lineno=114)

if os.name != 'posix':
    import pyttsx3
    engine = pyttsx3.init()

def speak(text):
    if os.name == 'posix':
        clean_text = ''.join(c for c in text if c.isalnum() or c.isspace() or c in ".,?!-_$:+-/")
        system(f"say '{clean_text}'")
    else:
        engine.say(text)
        engine.runAndWait()

def listen_for_wake_word(audio):
    global listening_for_wake_word
    # Add wake word detection logic here

def prompt_gpt(audio):
    global listening_for_wake_word
    result = base_model.transcribe(audio.get_raw_data())
    prompt_text = result['text']
    if not prompt_text.strip():
        print("Empty prompt. Please speak again.")
        speak("Empty prompt. Please speak again.")
        listening_for_wake_word = True
    else:
        print('User:', prompt_text)
        # Use the ChatGPT API to generate a response
        response = openai.Completion.create(
            engine="davinci",
            prompt=prompt_text,
            max_tokens=50
        )
        output = response.choices[0].text.strip()
        print('GPT4All:', output)
        speak(output)
        print(f'\nSay {wake_word} to wake me up.\n')
        listening_for_wake_word = True

def callback(recognizer, audio):
    global listening_for_wake_word
    if listening_for_wake_word:
        listen_for_wake_word(audio)
    else:
        prompt_gpt(audio)

def start_listening():
    with source as s:
        r.adjust_for_ambient_noise(s, duration=2)
    print(f'\nSay {wake_word} to wake me up.\n')
    r.listen_in_background(source, callback)

if __name__ == '__main__':
    r = sr.Recognizer()  # Initialize recognizer here
    start_listening()
    while True:
        time.sleep(1)
