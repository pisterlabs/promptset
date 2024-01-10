import elevenlabs
from openai import OpenAI, AsyncOpenAI
from elevenlabs import generate, play, set_api_key, voices, stream
import requests
import os

set_api_key(os.environ.get('ELEVENLABS_API_KEY'))


class LLMInterface:
    def get_response(self, system_messages: list, user_messages: str):
        raise NotImplementedError

    def get_response_image(self, user_id: int, instruction: str, image_url: str):
        raise NotImplementedError

    def get_response_image_stream(self, user_id: int, instruction: str, image_url: str):
        raise NotImplementedError


class AudioInterface:
    def get_audio(self, narrator: str, text: str):
        raise NotImplementedError


class OpenAIAPI(LLMInterface):
    def __init__(self, user_repo):
        self.openai_client = OpenAI()
        self.user_repo = user_repo

    def get_response(self, system_messages: list, user_message: str) -> str:
        messages = self._build_messages(system_messages, user_message)
        response = self.openai_client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=messages
        )
        return response.choices[0].message.content

    @staticmethod
    def _build_messages(system_messages: list, user_message: str = None) -> list:
        messages = [{'role': 'system', 'content': message} for message in system_messages]

        if user_message:
            messages.append({'role': 'user', 'content': user_message})

        return messages

    def get_response_image(self, user_id: int, instruction: str, image_url: str) -> str:
        context = self.user_repo.get_user_context(user_id)
        messages = self._build_messages_image(instruction, image_url, context)
        response = self.openai_client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=messages,
            max_tokens=400
        )
        return response.choices[0].message.content

    def get_response_image_stream(self, user_id: int, instruction: str, image_url: str):
        context = self.user_repo.get_user_context(user_id)
        messages = self._build_messages_image(instruction, image_url, context)
        response = self.openai_client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=messages,
            max_tokens=300,
            stream=True
        )

        response_chunks = []
        for event in response:
            current_response = event.choices[0].delta.content
            if current_response:
                response_chunks.append(current_response)
                yield current_response

        self.user_repo.add_user_context(user_id, ''.join(response_chunks))

    @staticmethod
    def _build_messages_image(instruction: str, image_url: str, context: list) -> list:
        return ([{'role': 'system', 'content': instruction}] +
                [{'role': 'assistant', 'content': context_piece} for context_piece in context] +
                [
                    {'role': 'user',
                     'content': [
                         {'type': 'text', 'text': "Narrate this image."},
                         {'type': 'image_url',
                          'image_url': {
                              'url': image_url,
                          }}
                     ]}
                ])


class ElevenLabsAPI(AudioInterface):
    def __init__(self):
        self.narrator_voice_ids = {
            'Sir David Attenborough': 'gKt3j6pa3jCxedjH1ZXk',
            'Morgan Freeman': 'Sq3XsrgFWr3VLEqCrq42'
        }

    def get_audio(self, narrator: str, text: str):
        elevenlabs_stream = elevenlabs.generate(
            text=text,
            voice=self.narrator_voice_ids[narrator],
            stream=True
        )

        yield from elevenlabs_stream
