import configparser
import logging
from openai import OpenAI
from core.pushtotalk import PushToTalk

class VoiceSynthesizer(logging.Handler):
    def __init__(self, synthesizer="openai", language="spanish"):
        self.synthesizer = synthesizer
        self.language = language
        self.language_argument = self._set_language_argument(language)
        self.text_to_speech_argument = self._set_text_to_speech_argument()

    def set_synthesizer(self, synthesizer):
        self.synthesizer = synthesizer

    def set_language(self, language):
        self.language = language
        self.language_argument = self._set_language_argument()
        self.text_to_speech_argument = self._set_text_to_speech_argument()

    def _set_language_argument(self, language):
        if self.synthesizer == "festival":
            return "--language " + self.language
        elif self.synthesizer == "espeak":
            return "-v en" if self.language == "english" else "-v es-la"
        elif self.synthesizer == "google":
            return "en" if self.language == "english" else "es"

    def _set_text_to_speech_argument(self):
        if self.synthesizer == "festival":
            return "--tts"
        elif self.synthesizer == "espeak":
            return "--stdout"

    def convert_to_audio(self, text, output_speech_file):

        services = configparser.ConfigParser()
        path = "configuration/services.config"
        services.read(path)
        api_key = services.get("openai", "api_key")

        if self.synthesizer == "openai":
            client = OpenAI(api_key=api_key)
            response = client.audio.speech.create(
                model="tts-1",
                voice="nova",
                input=text
            )
            return response.stream_to_file(output_speech_file)

    def speech_it(self, text):
        logging.info(text)
        text = "\"" + text + "\""
        pushtotalk = PushToTalk()
        _resource_type = "text"

        if self.synthesizer == "festival":
            command = f'echo {text} | {self.synthesizer} {self.text_to_speech_argument} {self.language_argument}'
        elif self.synthesizer == "espeak":
            command = f'{self.synthesizer} {self.language_argument} {self.text_to_speech_argument} {text} | aplay'
        elif self.synthesizer == "google":
            command = f'core/google.sh {self.language} {text}'
            # Or use: command = f'core/voicerss.sh {self.language} {text}'
        elif self.synthesizer == "openai":
            output_speech_file = "/tmp/audio.mp3"
            _resource_type = "audio"
            self.convert_to_audio(text, output_speech_file)

        pushtotalk.message(_resource_type, output_speech_file)
