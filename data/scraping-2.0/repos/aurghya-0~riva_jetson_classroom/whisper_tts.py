from pathlib import Path
from openai import OpenAI
from openai_client import OpenAIKey

# OpenAI Client Setup
client = OpenAI(api_key=OpenAIKey.key)

# Open or create the file for reading the transcript 
with open("text.txt", encoding="utf8") as file:
    input_text = file.read()


speech_file_path = Path(__file__).parent / "speech.mp3"
response = client.audio.speech.create(
  model="tts-1",
  voice="alloy",
  input=input_text
)

response.stream_to_file(speech_file_path)