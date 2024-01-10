from openai import OpenAI
import sounddevice as sd
from scipy.io.wavfile import write
import config

client = OpenAI(api_key = config.key)

def record_and_transcribe(filename="output.mp3", duration=5, sample_rate=44100):
    print("Recording...")

    # Record audio in monaural (single channel)
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()

    print("Recording complete.")

    # Save audio to a WAV file
    write(filename, sample_rate, audio_data)

    audio_file= open(filename, "rb")
    transcript = client.audio.transcriptions.create(
      model="whisper-1", 
      file=audio_file
    )

    return transcript.text

# Example usage
if __name__ == "__main__":
    file_name = "output.mp3"
    msg = record_and_transcribe(file_name)
    print()








