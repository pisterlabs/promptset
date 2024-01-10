# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
import openai
import os
from dotenv import load_dotenv

# Initialize OpenAI API Key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Set the OpenAI API key
openai.api_key = openai_api_key

def transcribe_audio(audio_file_path):
    with open(audio_file_path, 'rb') as audio_file:
        transcription = openai.Audio.transcribe("whisper-1", audio_file)
    return transcription['text']

audio_file_path = "audio/Eurostar (feat. Central Cee).mp3"
transcript = transcribe_audio(audio_file_path)

# Save the transcript to a text file
with open('transcript.txt', 'w') as f:
    f.write(transcript)