import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(grandparent_dir)

from configurations.config import OPENAI_API_KEY
from configurations.config_tts_stt import PATH_TO_SAVE
from openai import OpenAI

class T2S_with_openai():
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.text_to_speech_model = self.client.audio.speech
        self.model = "tts-1"
        self.voice = "shimmer"
    
    def create(self, input_, path_to_save = PATH_TO_SAVE):
        response = self.text_to_speech_model.create(
          model=self.model, 
          voice=self.voice,
          input=input_)
        path_file = os.path.join(path_to_save , "output.mp3")
        response.stream_to_file(path_file)
        
        return path_file

