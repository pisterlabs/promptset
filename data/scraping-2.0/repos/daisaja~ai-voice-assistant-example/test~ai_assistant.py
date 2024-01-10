import os
import time
import pyaudio
import playsound
from gtts import gTTS
import openai
import speech_recognition as sr  # SpeechRecognition

api_key = "YOUR_CHATGPT_API_KEY"

lang = 'en'
openai.api_key = api_key

while True:
    def get_audio():
        r = sr.Recognizer()
        with sr.Microphone() as source: # (device_index=2)
            audio = r.listen(source)
            said = ""

            try:
                said = r.recognize_google(audio)
                print(said)

                if "Computer" in said:
                    completion = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "user", "content": said}
                        ]
                    )
                    text = completion.choices[0].message.content
                    speech = gTTS(text=text, lang=lang, slow=False, tld="co.uk")
                    speech.save("welcome1.mp3")
                    playsound.playsound("welcome1.mp3")
            except Exception as e:
                print("Exception: " + str(e))

        return said

    get_audio()