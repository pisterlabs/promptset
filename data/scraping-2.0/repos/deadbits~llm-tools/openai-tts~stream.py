import io
from openai import OpenAI
from pydub import AudioSegment
from pydub.playback import play

client = OpenAI()

def stream_and_play(text):
  response = client.audio.speech.create(
    model="tts-1",
    voice="alloy",
    input=text,
  )

  # Convert the binary response content to a byte stream
  byte_stream = io.BytesIO(response.content)

  # Read the audio data from the byte stream
  audio = AudioSegment.from_file(byte_stream, format="mp3")

  # Play the audio
  play(audio)


if __name__ == "__main__":
  text = input("Enter text: ")
  stream_and_play(text)
