import os
import dotenv

from pathlib import Path
from openai import OpenAI

dotenv.load_dotenv()

HERE = Path(__file__).parent

API_KEY = os.getenv("OPENAI_API_KEY", None)

if API_KEY is None:
    raise ValueError("OPENAI_API_KEY not set")
else:
    print(API_KEY)

client = OpenAI(api_key=API_KEY)

# print(client.models.list())

text = input("What do you want me to voice: ")

speech_file_path = HERE / "speech.mp3"
response = client.audio.speech.create(
  model="tts-1",
  voice="alloy",
 input=f"{text}"
)

speech_file_path = HERE / "speech.mp3"

response.stream_to_file(speech_file_path)

# play the file with afplay
os.system(f"afplay {speech_file_path}")
