import os
from typing import Iterable
from openai import OpenAI

class Synthesizer:
  def __init__(self, client: OpenAI | None = None):
    self.client = client or OpenAI(
      api_key=os.environ.get("OPENAI_API_KEY"),
    )

  def synthesize(self, text) -> Iterable[bytes]:
    response = self.client.audio.speech.create(
      model='tts-1',
      voice='alloy',
      input=text,
    )

    return response.iter_bytes()
