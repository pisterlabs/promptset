import abc
import io

import openai
import requests


class TTSHelper(abc.ABC):
    @abc.abstractmethod
    def text_to_speech(self):
        raise NotImplementedError

class Custom_TTSHelper(TTSHelper):
    def __init__(self, endpoint):
        self.endpoint = endpoint

    def text_to_wav(self, message:str, params:dict={"lengthScale":"0.65"}):
        if not message or len(message) == 0:
            return

        response = requests.post(self.endpoint, data=message, params=params)
        return io.BytesIO(response.content)
    
    def text_to_speech(self, *args, **kwargs):
        return self.text_to_wav(*args, **kwargs)
    
class OpenAI_TTSHelper(TTSHelper):
    def __init__(self, client):
        self.client = client
        if not self.client:
            self.client = openai.OpenAI()

    def text_to_mp3(self, message:str):
        if not message or len(message) == 0:
            return

        response = self.client.audio.speech.create(
            model="tts-1",
            voice="nova",
            input=message
        )
        
        return io.BytesIO(response.content)
    
    def text_to_speech(self, *args, **kwargs):
        return self.text_to_mp3(*args, **kwargs)