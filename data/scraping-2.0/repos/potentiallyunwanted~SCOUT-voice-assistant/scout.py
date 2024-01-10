from threading import Thread
import openai
from elevenlabs import generate, play, set_api_key
import os
import speech_recognition as sr

class Scout(Thread):
    def __init__(self, voice_name="roboBrit"):
        super().__init__()
        self.voice_name = voice_name
        self.set_api_keys()
    
    def set_api_keys(self):
        try:
            elevenlabs_api_key= os.environ["ELEVENLABS_API_KEY"]
            set_api_key(elevenlabs_api_key)
            openai.api_key = os.environ["OPENAI_API_KEY"]
        except KeyError as e:
            print(f"Environment variable {str(e)} not found.")
            exit(1)

    def start(self):
        while True:
            human = self.get_human_speech()
            if human is not None:
                ai = self.get_ai_response(human)
                self.get_voice(ai)

    def get_human_speech(self):
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("Please say something...")
            try:
                audio = r.listen(source, timeout=5)  # Listen for a max of 5 second
                return sr.whisper.recognize_whisper_api(audio_data=audio, recognizer=r)
            except sr.WaitTimeoutError:
                print("No speech detected within the timeout period.")
            except sr.UnknownValueError:
                print("I'm sorry, I didn't catch that.")
            except sr.RequestError as e:
                print("Could not request results from Google Speech Recognition service; {0}".format(e))
        return None

    def get_ai_response(self, input):
        chat_completion = openai.ChatCompletion.create(model="gpt-4", messages=[{"role": "user", "content": input}])
        return chat_completion.choices[0].message.content

    def get_voice(self, text):
        audio = generate(text=text, voice=self.voice_name)
        play(audio)
    