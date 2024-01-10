import openai
import os
from dotenv import load_dotenv

if __name__ == "__main__":
    load_dotenv()  # Read local .env file

openai.api_key = os.environ['OPENAI_API_KEY']
audio_file= open("data/audio.mp3", "rb")
transcript = openai.Audio.transcribe("whisper-1", audio_file, prompt="interest rate, loan, gold")
print(transcript)
