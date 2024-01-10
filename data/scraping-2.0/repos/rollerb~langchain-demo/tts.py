import io, soundfile as sf, sounddevice as sd
from openai import OpenAI

def play_text(text: str) -> None:
  openai = OpenAI()
  spoken_response = openai.audio.speech.create(
    model="tts-1",
    voice="nova",
    response_format="opus",
    input=text
  )
  buffer = io.BytesIO()
  for chunk in spoken_response.iter_bytes(chunk_size=4096):
    buffer.write(chunk)
  buffer.seek(0)
  
  with sf.SoundFile(buffer, 'r') as sound_file:
    data = sound_file.read(dtype="int16")
    sd.play(data, sound_file.samplerate)
    sd.wait()