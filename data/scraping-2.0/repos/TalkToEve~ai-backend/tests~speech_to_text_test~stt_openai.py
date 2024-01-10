import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(grandparent_dir)

from configurations.config import OPENAI_API_KEY
from openai import OpenAI

client = OpenAI(api_key=OPENAI_API_KEY)

audio_file = open(current_dir + "/" + "output.mp3", "rb") # "Hello world! This is a streaming test."

transcript = client.audio.translations.create(
  model="whisper-1", 
  file=audio_file
)
print(transcript)