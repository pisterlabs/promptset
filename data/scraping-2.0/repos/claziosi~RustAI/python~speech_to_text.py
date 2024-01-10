import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI()

# Load your API key from an environment variable or secret management service
client.api_key = os.getenv("OPENAI_API_KEY");

file = "./media/question.m4a"
audio_file= open(file, "rb")
transcript = client.audio.transcriptions.create(
  model="whisper-1", 
  file=audio_file,
  language="en"
)

print(transcript.text)