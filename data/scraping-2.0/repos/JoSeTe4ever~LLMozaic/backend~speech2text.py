import dotenv
import os
import openai

dotenv.load_dotenv()
OPEN_API_KEY = os.getenv("OPEN_API_KEY");

def transcribe(filepath):
    print(f"Valor del parámetro 'filepath': {filepath}")
    audio = open(filepath, 'rb')
    openai.api_key = OPEN_API_KEY
    speech2text = openai.Audio.transcribe(model="whisper-1", file=audio, speaker_labels=True)
    print(f"Valor del parámetro 'speech2text': {speech2text}")
    return speech2text["text"];