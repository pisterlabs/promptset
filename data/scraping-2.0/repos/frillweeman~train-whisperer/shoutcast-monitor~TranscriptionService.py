from openai import OpenAI
from os import path
import re
import configparser

config = configparser.ConfigParser()
config.read("config.ini")

class TranscriptionService:

  def __init__(self, openai_api_key):
    self.__client = OpenAI(api_key=openai_api_key)

  def __postprocess(self, text):
    return [t for t in re.split(r"\s*»»\s*", text) if t.strip()]

  def transcribe(self, filename):
    afpath = path.join(path.dirname(path.realpath(__file__)), filename)
    audio_file= open(afpath, "rb")
    transcript = self.__client.audio.transcriptions.create(
      model="whisper-1", 
      file=audio_file,
      language="en",
      prompt=config["transcription_service"]["prompt"],
    )
    return self.__postprocess(transcript.text)
