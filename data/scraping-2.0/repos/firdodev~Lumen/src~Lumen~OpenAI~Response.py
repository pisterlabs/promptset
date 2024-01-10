import openai
import time
import os
import pyttsx3
import pygame
from Lumen.Resources.ConfigLoader import LoadConfig

class OpenAIResponse:
    def __init__(self):
        openai.api_key = LoadConfig().get('api', '')
        self.model_engine = "text-davinci-003"
        self.base_prompt = "Hello! My name is Lumen. How can I assist you today? "

        # Initialize pygame for audio playback

        self.last_response = ""
        pygame.init()
        pygame.mixer.init()

    def text_to_speech(self, text):
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')

        # You can set different properties here
        engine.setProperty('voice', voices[1].id)  # Change the index to switch between voices
        engine.setProperty('rate', 180)  # Speed of speech

        engine.say(text)
        engine.runAndWait()

    def play_audio(self, audio_path):
        # Play the generated audio
        pygame.mixer.music.load(audio_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        os.remove(audio_path)  # delete the temporary file after playing

    def get_response(self, prompt, retries=3, delay=10):
        for _ in range(retries):
            try:
                response = openai.Completion.create(
                    engine=self.model_engine,
                    prompt=prompt,
                    max_tokens=1024,
                    n = 1,
                    stop = None, 
                    temperature = 0.5,
                )
                return response.choices[0].text  # Removing any leading or trailing spaces
            except openai.error.RateLimitError:
                print(f"Rate limit exceeded. Retrying in {delay} seconds...")
                time.sleep(delay)
        raise Exception("Exceeded maximum retries due to rate limiting.")

    def interactive_mode(self, prompt):
        user_input = prompt
        if user_input.lower() in ["quit", "exit"]:
            print("Goodbye!")
            return "Goodbye!"
        response = self.get_response(self.base_prompt + user_input, delay = 5)
        print("Lumen:", response)
        self.last_response = response
        # Convert response to speech and play
        audio_path = self.text_to_speech(response)
        self.play_audio(audio_path)
