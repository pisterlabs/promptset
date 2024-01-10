import pyttsx3
import speech_recognition as sr
import os
import openai


api_key = os.environ['OpenAI']


class ChatBot:
    def __init__(self):
        self.engine=pyttsx3.init('sapi5')
        self.voices=self.engine.getProperty('voices')
        self.engine.setProperty('voice',self.voices[0].id)
        openai.api_key = api_key
        self.r=sr.Recognizer() 
    def speak(self,audio):
        self.engine.say(audio)
        self.engine.runAndWait()
    def close_speak(self):
        self.engine.stop()
    def take_commands(self):
        with sr.Microphone() as Source:
            print("Listening...")
            self.r.pause_threshold=1
            audio=self.r.listen(source = Source, timeout= None, phrase_time_limit= 5)
        try:
            print("Recognising...")
            command=self.r.recognize_google(audio,language='en-in')
            print(f"User asked for  {command}")
            return command
        except Exception as e:
            self.speak("Say That Again Please..")
            command = self.take_commands()
            return command

    def get_response(self, user_input):
        response = openai.Completion.create(
            engine = "text-davinci-003",
            prompt = user_input,
            max_tokens= 4000,
            temperature = 0.5
        ).choices[0].text
        return response
    
# if __name__ == "__main__":
#     chatbot = ChatBot()
#     response = chatbot.get_response("Tell me about Wipro")
#     print(response)