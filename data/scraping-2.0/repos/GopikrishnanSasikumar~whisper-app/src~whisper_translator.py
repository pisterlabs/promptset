from openai import OpenAI
import dotenv
import os

dotenv.load_dotenv("ops/.env")

def translate_audio(file_path, model="whisper-1"):
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    audio_file = open(file_path, "rb")
    transcript = client.audio.translations.create(
        model=model,
        file=audio_file
    )
    transcript = transcript.text
    return transcript
