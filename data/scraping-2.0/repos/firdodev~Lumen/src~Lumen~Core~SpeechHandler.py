import speech_recognition as sr
import json
import os

from Lumen.Resources.ConfigLoader import LoadConfig

from Lumen.OpenAI.Response import OpenAIResponse


class SpeechHandler:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 3000

        self.activation_word = LoadConfig().get('name', '').lower()
        self._last_sentence = ""
        self._ai_response = ""
        self.response = OpenAIResponse()
        self.activated = False

    def check_activation_word(self, raw_audio_data):
        print("Listening...")
        try:
            with sr.Microphone() as source2:
                self.recognizer.adjust_for_ambient_noise(source2, duration=0.2)
                audio = self.recognizer.listen(source2, timeout=5, phrase_time_limit=5)

                detected_text = self.recognizer.recognize_google(audio)
                self._last_sentence = ""
                self._last_sentence = detected_text 

                if self.activated:
                    self.response.interactive_mode(detected_text)

                if self.activation_word in detected_text.lower():
                    self.activated = True
                    return True
                return False
        except sr.UnknownValueError:
            print("Audio not understood or too noisy.")
            return False
        except Exception as e:
            print("Error:", e)
            return False

    def get_last_sentence(self):
        """
        Returns the most recent sentence recognized by the SpeechHandler.

        Returns:
        - A string containing the last recognized sentence. Empty string if no sentence has been recognized yet.
        """
        return self._last_sentence
    
    def get_lumen_response(self):
        return self._ai_response
