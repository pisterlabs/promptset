import os

from openai import OpenAI

from ....config import get_settings

settings = get_settings()
os.environ["OPENAI_API_KEY"] = settings.openai_api_key

client = OpenAI()


async def handle_text_to_speech(text: str):
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text,
    )
    return response
