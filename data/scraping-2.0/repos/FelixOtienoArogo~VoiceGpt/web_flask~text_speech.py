from pathlib import Path
from openai import OpenAI
import os
client = OpenAI()

def text_speech(text):
    #Getting the file path
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) #path of this file
    REPLIES_DIR = os.path.join(SCRIPT_DIR, 'replies')
    os.makedirs(REPLIES_DIR, exist_ok=True)
    mp3_file_path = os.path.join(REPLIES_DIR, 'reply.mp3')

    speech_file_path = mp3_file_path
    response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text
            )
    response.stream_to_file(speech_file_path)

    return speech_file_path
