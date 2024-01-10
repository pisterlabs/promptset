from pathlib import Path
from openai import OpenAI
client = OpenAI()

speech_file_path = Path(__file__).parent / "speech_eng.mp3"
response = client.audio.speech.create(
  model="tts-1",
  voice="onyx",
  input="A fire has been detected, evacuate immediately!"
)

response.stream_to_file(speech_file_path)