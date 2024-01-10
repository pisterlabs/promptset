import json
import requests
import openai

from configuration.config import Configuration
from utility.speech_to_text_converter.SpeechToTextGenerator import SpeechToTextGenerator
from utility.text_to_speech_converter.TextToSpeechGenerator import TextToSpeechGenerator


class RequestHandler:

    def __init__(self):
        self.response = None
        self.prompt = None
        self.temperature = 0.2
        self.response_message = ''
        openai.api_key = Configuration.OPENAI_API_KEY

    def get_prompt_from_user(self, last_prompt):
        speech_obj = SpeechToTextGenerator()
        speech_obj.listen_for_speech()
        prompt = last_prompt + speech_obj.get_text()
        self.__send_prompt(prompt)

    def __send_prompt(self, prompt):
        self.prompt = prompt
        # self.__send_request_to_openai_model()
        self.__send_stream_request_to_openai_model()

    def play_response_as_a_speech(self):
        if self.response_message is not None:
            converted_obj = TextToSpeechGenerator(self.response_message)
            converted_obj.play_speech()

    @staticmethod
    def play_response_stream_as_a_speech(streamed_text):
        converted_obj = TextToSpeechGenerator(streamed_text)
        converted_obj.play_speech()

    def get_response_as_a_transcript(self):
        if self.response_message is not None:
            return self.response_message

    def __send_request_to_openai_model(self):
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=[
                {"role": "user", "content": self.prompt}
            ],
            temperature=self.temperature
        )
        self.response = response.to_dict()
        self.response_message = self.response.get('choices')[0].get('message').get('content')
        self.play_response_as_a_speech()

    def __send_stream_request_to_openai_model(self):
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=[
                {"role": "user", "content": self.prompt}
            ],
            temperature=self.temperature,
            stream=True
        )

        limit = 10
        chunk_at = 0
        subtext = ''
        for chunk in response:
            chunk_content = chunk.get('choices')[0].get('delta').get('content', None)
            if chunk_content is None:
                continue
            subtext += chunk_content
            if chunk_at >= limit:
                self.response_message += subtext
                self.play_response_stream_as_a_speech(subtext)
                subtext = ''
                chunk_at = 0
            chunk_at += 1
        if subtext != '':
            self.response_message += subtext
            self.play_response_stream_as_a_speech(subtext)
