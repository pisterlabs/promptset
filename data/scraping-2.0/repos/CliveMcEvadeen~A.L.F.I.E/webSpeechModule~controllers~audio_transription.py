import requests
import base64
from openai import OpenAI
import os 
from dotenv import load_dotenv

load_dotenv()

open_ai_api_key = os.getenv("open_ai_api_key")


client = OpenAI(api_key=open_ai_api_key)


def transcribe_audio(audio_path, transcriptions):
    audio_file = open(audio_path, "rb")
    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        prompt=fr"""
        The following is part of live speech going on, try to make as mich sence of it as possible, some sounds might not be clear but try to make as much sence of it and return the most accurate transcript you can come up with.
        
        # This is a list of the previous transcriptions that you have made from the same user. use them as a reference to make the current transcription more accurate.

        {transcriptions}

        # Note: The transcriptions list can be empty at the beginning of the conversation.

        """,
        response_format="json",
        file=audio_file,

    )
    
    return transcription

# load and test the with a file.

