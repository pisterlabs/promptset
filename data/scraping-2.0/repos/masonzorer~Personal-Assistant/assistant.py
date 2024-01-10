# Project for speech to text to GPT response
from dotenv import load_dotenv
import os
import speech_recognition as sr
from gtts import gTTS
from playsound import playsound
import openai

# Function to convert text to speech
def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    tts.save("audio.mp3")
    playsound("audio.mp3")

# Function to get assistant response
def get_response(audio):
    response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=audio,
                max_tokens=100,
            )
    if response.choices:
        text_to_speech(response.choices[0].text.strip())
    return None

# listen for input from user
def listen_for_input(r, mic, timeout):
    with mic as source:
        audio = r.listen(source, phrase_time_limit=timeout)
        r.adjust_for_ambient_noise(source)
        try:
            audio = r.recognize_google(audio)
            return audio
        except sr.UnknownValueError:
            print("Could not understand audio")

def main():
    # initialize gpt engine
    load_dotenv()
    API_key = os.getenv("OpenAI_API_KEY")
    openai.api_key = API_key

    # initialize speech recognition engine
    r = sr.Recognizer()
    mic = sr.Microphone()

    # Loop for assistant
    while True:

        # wait for assistant to be called
        sleeping = True
        while sleeping:
            input = listen_for_input(r, mic, 1)
            if input:
                if 'emma' in input.lower():
                    sleeping = False
                    text_to_speech("Yes?, How can I help you?")
                    break

        # listen for command and respond
        awake = True
        while awake:
            input = listen_for_input(r, mic, 3)
            if input:
                if 'no' in input.lower():
                    text_to_speech("Okay, goodbye.")
                    awake = False
                    break
                else:
                    get_response(input)
                    text_to_speech("Anything else?")
            else:
                text_to_speech("I didn't hear anything, goodbye.")
                awake = False
                break

if __name__ == "__main__":
    main()