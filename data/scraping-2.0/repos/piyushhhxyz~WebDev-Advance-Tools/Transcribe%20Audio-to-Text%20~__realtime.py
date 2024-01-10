import sounddevice as sd
import numpy as np
import openai
import io

# Set up OpenAI API key
openai.api_key = 'YOUR_OPENAI_API_KEY'

# Callback function to process audio chunks
def audio_callback(indata, frames, time, status):
    volume_norm = np.linalg.norm(indata) * 10
    if volume_norm > 0.5:
        audio_buffer = io.BytesIO()
        audio_buffer.write(indata.tobytes())
        audio_buffer.seek(0)
        
        try:
            response = openai.Audio.transcriptions.create(
                file=audio_buffer,
                model="whisper-1"
            )
            print(response["text"])
        except Exception as e:
            print(f"Error: {e}")

# Start recording from the microphone
with sd.InputStream(callback=audio_callback):
    print("Recording... Press Ctrl+C to stop.")
    sd.sleep(1000000)
