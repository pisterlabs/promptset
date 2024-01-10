import os
from typing import Optional

from openai import OpenAI

from dosaku import Config, Service
from dosaku.types import Audio
from dosaku.utils import ifnone


class OpenAITextToSpeech(Service):
    name = 'OpenAITextToSpeech'
    config = Config()

    def __init__(self):
        super().__init__()
        self.client = OpenAI(api_key=self.config['API_KEYS']['OPENAI'])
        self.model = 'tts-1'
        self.voices = ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer']
        self.voice = 'alloy'

    def set_voice(self, voice: str):
        self.voice = voice

    def text_to_speech(self, text: str, output_filename: Optional[str] = None, voice: Optional[str] = None) -> Audio:
        voice = ifnone(voice, default=self.voice)
        response = self.client.audio.speech.create(
            model=self.model,
            voice=voice,
            input=text
        )

        if output_filename is not None:
            response.stream_to_file(output_filename)
            audio = Audio(filename=output_filename)
        else:
            output_filename = os.path.join(self.config['DIR_PATHS']['TEMP'], 'audio.mp3')
            response.stream_to_file(output_filename)
            audio = Audio(filename=output_filename)

        return audio


OpenAITextToSpeech.register_action('text_to_speech')
