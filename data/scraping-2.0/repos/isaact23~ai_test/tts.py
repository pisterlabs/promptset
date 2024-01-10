from pathlib import Path
from openai import OpenAI

client = OpenAI()

audio_path = Path(__file__).parent / "echo2.mp3"
response = client.audio.speech.create(
  model="tts-1",
  voice="echo",
  input="Bitch your plan was absolute cheeks"
)

response.stream_to_file(audio_path)
