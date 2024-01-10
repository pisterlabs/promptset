import speech_recognition as sr
import openai
import config
from gtts import gTTS
import os

def speech_to_text(duration=5, language="en"):
    r = sr.Recognizer()

    while True:
        with sr.Microphone() as source:
            # read the audio data from the default microphone
            print('start recognizing')
            audio_data = r.record(source, duration=duration)
            print("Recognizing...")
            try:
                # convert speech to text
                text = r.recognize_google(audio_data, language=language)
                print("You said:", text)
                return text
            except sr.UnknownValueError:
                print("Could not understand audio.")
                continue
            except sr.RequestError as e:
                print(f"Error occurred during speech recognition: {str(e)}")
                continue

def GPT_Completion(texts):
    # Call the API key under your account (in a secure way)
    openai.api_key = config.OPEN_API_KEY
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=texts,
        temperature=0.6,
        top_p=1,
        max_tokens=600,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].text

def text_to_speech_gttps(mytext, lang="en", file_path=''):
    audio = gTTS(text=mytext, lang=lang, slow=False)
    audio.save(file_path)
    os.system("start " + file_path)

language = "en"
duration = 5

while True:
    in_text = speech_to_text(duration=duration, language=language)
    if in_text:
        if in_text.lower() in ["stop", "exit"]:
            break
        out_text = GPT_Completion(in_text)
        print("Response:", out_text)
        text_to_speech_gttps(out_text, lang=language, file_path="response.mp3")
