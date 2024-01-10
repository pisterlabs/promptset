import os
from sys import exit
import openai
from dotenv import load_dotenv

def transcribe_openai():
    load_dotenv()
    try:
        openai.api_key = os.getenv("CHATGPT_API")
        audio_file = open("rec.wav", "rb")
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
        return transcript["text"]
    except:
        print("check the openai key")
        exit(0)

