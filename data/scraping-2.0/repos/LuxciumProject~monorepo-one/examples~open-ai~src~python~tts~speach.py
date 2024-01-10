import os
import openai
from pydub import AudioSegment
from pydub.playback import play
import io

# Set API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Create OpenAI client
client = openai.OpenAI()

# Generate speech audio
response = client.audio.speech.create(
    model="tts-1",
    voice="alloy",
    input="Hello world! This is a streaming test."
)

# Stream audio to speakers
audio_stream = io.BytesIO(response.content)
song = AudioSegment.from_file(audio_stream, format="mp3")
play(song)
