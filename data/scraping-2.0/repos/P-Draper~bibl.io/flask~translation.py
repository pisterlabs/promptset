import os
from pymongo import MongoClient
from dotenv import load_dotenv
import openai
import io
import codecs
import json
import ast

load_dotenv()
DATABASE = os.environ.get("DB_URI", f"sqlite:///{os.path.abspath(os.path.dirname(__file__))}/app.db")
MONGO_URI = os.environ.get("MONGO", '')
client = MongoClient(MONGO_URI)
db = client.Test

OPENAI_WHISPER_API_KEY = os.environ.get("OPENAI_WHISPER_API_KEY")
openai.api_key = OPENAI_WHISPER_API_KEY


class CustomAudioFile:
    def __init__(self, data, name):
        self.data = data
        self.name = name
    def read(self):
        return self.data

    def __len__(self):
        return len(self.data)


def translate_audio(audio_data_bytes):
    try:
        print("Type of audio_data_bytes:", type(audio_data_bytes))

        model_id = 'whisper-1'
        translation = openai.Audio.translate(
            model=model_id,
            file=CustomAudioFile(audio_data_bytes, "temp_audio.mp3"),
            content_type='audio/mp3',
            response_format='json'
        )
        
        print("translation result:", translation)

        if translation.get("error"):
            print(f"OpenAI Error: {translation['error']['message']}")
            return None

        try:
            decoded_text = ast.literal_eval(f'"{translation["text"]}"')
        except (SyntaxError, ValueError):
            decoded_text = translation['text']

        return decoded_text
    except Exception as e:
        print(f"Error during translation: {str(e)}")
        return None
