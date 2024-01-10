import os
import sys
import logging
import yaml
import openai
import speech_recognition as sr
from gtts import gTTS
from subprocess import run
from UserInput import UserInput
from openai_module import OpenAI
from AudioProcessor import AudioProcessor

class Sabia:
    def __init__(self, config_file):
        self.config = self._load_config(config_file)
        self.logger = self._setup_logger()
        self.audio_processor = AudioProcessor()
        self.voice_recognition = VoiceRecognition()
        self.response_generator = ResponseGenerator()
        self.text_to_speech = TextToSpeech()
        
    def start(self):
        self.logger.info("Sabia is now running. Press Ctrl+C to exit.")
        while True:
            try:
                audio = self.voice_recognition.listen()
                if audio:
                    preprocessed_audio = self.audio_processor.preprocess_audio(audio)
                    user_input = self.voice_recognition.convert_to_text(preprocessed_audio)
                    self.logger.info(f"User input: {user_input}")
                    response = self.response_generator.generate_response(user_input)
                    self.text_to_speech.convert_text_to_audio(response)
            except KeyboardInterrupt:
                self.logger.info("Sabia has been stopped.")
                break
                
    def _load_config(self, config_file):
        if not os.path.isfile(config_file):
            raise FileNotFoundError("Config file not found!")
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
        
    def _setup_logger(self):
        logger = logging.getLogger('Sabia')
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        return logger
