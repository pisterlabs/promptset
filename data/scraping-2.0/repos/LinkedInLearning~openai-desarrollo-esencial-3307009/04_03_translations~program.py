import os
from openai import OpenAI

key = os.environ.get("OPENAI_API_KEY")

client = OpenAI(api_key=key)

audio_file = open("openai_desarrollo_esencial.ogg", "rb")
transcription = client.audio.translations.create(
    model="whisper-1",
    response_format="text",
    file=audio_file
)

print(transcription)