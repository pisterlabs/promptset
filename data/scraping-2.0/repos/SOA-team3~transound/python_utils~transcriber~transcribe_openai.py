import os
# https://platform.openai.com/docs/guides/speech-to-text/supported-languages
from openai import OpenAI
import sys

# Get Ruby Input
ruby_input = sys.stdin.read()
audio_file_path = ruby_input.split('\n')[0]
openai_api_key = ruby_input.split('\n')[1]

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = openai_api_key

# Create an OpenAI client
client = OpenAI()

# Open the audio file
# audio_file_path = "podcast_mp3_store/Ricky Gervais: A Joke About His Will.mp3"
audio_file = open(audio_file_path, "rb")

# Create a transcription using the Whisper model
transcript = client.audio.transcriptions.create(
    model="whisper-1",
    file=audio_file,
    response_format="text"
)

# Print the generated transcript
print(transcript)