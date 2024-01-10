import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(".env")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

with open("frame_text.json", "r") as f:
    frames = json.load(f)


for frame in frames:
    frame_id = frame["frame_id"]
    frame_text = frame["text"]
    print(f"Processing Frame {frame_id}...")

    response = client.audio.speech.create(model="tts-1", voice="echo", input=frame_text)

    speech_file_path = f"narration/{frame_id}.mp3"
    response.stream_to_file(speech_file_path)
