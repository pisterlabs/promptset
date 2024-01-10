from openai import OpenAI
from dotenv import load_dotenv
import os


load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)
print("loaded env")

audio_file = open("speech_combined.mp3", "rb")
transcript = client.audio.transcriptions.create(
    model="whisper-1", file=audio_file, response_format="text"
)
print(transcript)
