import openai
import os
from openai.types.audio import Transcription

class WhisperAPI:
  def __init__(self) -> None:
    self.client = openai.Client(api_key=os.environ["OPENAI_API_KEY"])
  
  def recognize_voice(self, audio_file_path):
      # Read the audio file
      with open(audio_file_path, "rb") as audio_file:
        response = self.client.audio.transcriptions.create(model="whisper-1", file=audio_file)
        return response.text