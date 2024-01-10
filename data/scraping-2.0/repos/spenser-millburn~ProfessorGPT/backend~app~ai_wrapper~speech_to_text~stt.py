import os
from anyio import Path
import sounddevice as sd
import soundfile as sf
from openai import OpenAI

class SpeechToText:
    """
    SpeechToText Class Summary:
    
    The SpeechToText class provides methods for recording audio, transcribing audio to text using OpenAI's API.
    
    Methods:
    - __init__(self): Initializes the SpeechToText class.
    - record_audio(self, duration=5): Records audio for a specified duration.
    - transcribe_audio(self, audio_path): Transcribes audio from the provided path to text.
    - record_and_transcribe_audio(self): Records audio and transcribes it to text.
    """
    
    def __init__(self):
        self.client = OpenAI()
        self.output_prefix = Path(os.getcwd()) / "output"
        if not self.output_prefix.exists():
          os.mkdir(self.output_prefix)

    def record_audio(self, duration=5):
        sample_rate = 44100
        channels = 1
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels)
        sd.wait()  # Wait until the recording is finished
        output_file_path = self.output_prefix / "answer.wav"
        sf.write(output_file_path, audio, sample_rate)
        return output_file_path

    def transcribe_audio(self, audio_path):
        audio_file = open(audio_path, "rb")
        transcript = self.client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
        return transcript

    def record_and_transcribe_audio(self):
        return self.transcribe_audio(self.record_audio())
