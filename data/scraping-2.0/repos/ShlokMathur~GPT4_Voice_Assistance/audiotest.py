import openai
import pyaudio
import wave
from pydub import AudioSegment
import Whisper
import json

# Initialize OpenAI API
openai.api_key = "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"

# Define function to transcribe audio
def transcribe(audio_data):
    response = openai.Completion.create(
        engine="davinci",
        prompt=(f"Transcribe the following audio:\n\n{audio_data.decode()}"),
        temperature=0.7,
        max_tokens=60,
        n=1,
        stop=None,
    )
    return response.choices[0].text.strip()

# Record audio using PyAudio
audio = pyaudio.PyAudio()
stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
print("Recording...")
frames = []
for i in range(0, int(16000 / 1024 * 5)):
    data = stream.read(1024)
    frames.append(data)
print("Finished recording.")
stream.stop_stream()
stream.close()
audio.terminate()

# Save audio to file
with wave.open("temp_audio.wav", "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
    wf.setframerate(16000)
    wf.writeframes(b"".join(frames))

# Load audio file using PyDub
audio = AudioSegment.from_file("temp_audio.wav", format="wav")

# Transcribe audio using OpenAI
text = transcribe(audio.raw_data.decode())

# Print transcribed text
print("Transcribed text:")
print(text)

# Convert text to whisper
whisper = Whisper().say(text)

# Print whisper as JSON
print("Whisper:")
print(json.dumps(whisper))
