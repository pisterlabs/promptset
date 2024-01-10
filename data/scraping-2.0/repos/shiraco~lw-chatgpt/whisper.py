import os

import openai

from dotenv import load_dotenv
load_dotenv()

# Settings for OpenAI
openai.api_key = os.environ["OPENAI_API_KEY"]


def generate_response(file_path: str = "./data/speech.m4a") -> str:
    """Generate response from Whisper

    :param text: request text
    :return: generated text
    """
    response = openai.Audio.transcribe(
        file=open(file_path, "rb"),
        model="whisper-1",
    )
    res = response.text.strip()

    return res
