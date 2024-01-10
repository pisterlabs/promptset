import os
import langid  # language detection library
import speech_recognition as sr  # speech recognition library
import pyttsx3  # text-to-speech library
import pygame  # audio playback library
from gtts import gTTS  # Google text-to-speech API library
import openai  # OpenAI language model API library

class chat_talk:
    def __init__(self, len='en',api=None):
        self.api=api
        self.len = len

    def start(self):
        r = sr.Recognizer()  # initialize the speech recognizer
        with sr.Microphone() as source:  # use microphone as audio source
            r.adjust_for_ambient_noise(source)  # adjust noise levels for better recognition
            print("Please say something")
            audio = r.listen(source)  # listen for audio input
            print("Recognizing Now .... ")
            if self.len=='ar':  # if the selected language is Arabic
                try:
                    text = r.recognize_google(audio,language='ar')  # use Google speech recognition to transcribe audio to Arabic text
                    text_ = self.chat(text)  # use the OpenAI language model to generate a response
                    print("You have said \n" + text)

                except Exception as e:
                    print("Error :  " + str(e))
            elif self.len=='en':  # if the selected language is English
                try:
                    text = r.recognize_google(audio)  # use Google speech recognition to transcribe audio to English text
                    text_ = self.chat(text)  # use the OpenAI language model to generate a response
                    print("You have said \n" + text)

                except Exception as e:
                    print("Error :  " + str(e))


            path = "recorded.wav"
            with open(path, "wb") as f:
                f.write(audio.get_wav_data())  # write the audio data to a file

        langid.set_languages(['ar', 'en'])  # set the languages for the language detector
        lang_code, _ = langid.classify(text_)  # detect the language of the generated response
        if lang_code == 'ar':  # if the language of the response is Arabic
            self.text_ar_sound(text_)  # use the Google text-to-speech API to generate an Arabic audio file and play it

        else:  # if the language of the response is English
            self.text2sound(text_)  # use the text-to-speech library to generate an audio file and play it

        return text_

    def text2sound(self,text_):
        engine = pyttsx3.init()  # initialize the text-to-speech engine
        engine.setProperty('rate', 150)  # set the speech rate to 150 words per minute
        engine.say(text_)  # speak the generated response
        engine.runAndWait()  # wait for speech to finish before continuing execution

    def text_ar_sound(self,text):

        tts = gTTS(text=text, lang='ar')  # generate an Arabic audio file using the Google text-to-speech API
        audio_file = "temp_audio.mp3"
        tts.save(audio_file)  # save the audio file
        pygame.init()  # initialize the audio playback library
        sound_file = "temp_audio.mp3"
        pygame.mixer.music.load(sound_file)
        pygame.mixer.music.play()  # play the audio file
        while pygame.mixer.music.get_busy():  # wait for the audio playback to finish
            continue

        pygame.quit()  # quit the audio playback

    def chat(self, text):

        openai.api_key =self.api  # set the OpenAI API key

        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": text}
            ]
        )  # use the OpenAI language model to generate a response based on the user input
        text = completion.choices[0].message['content']
        return text
