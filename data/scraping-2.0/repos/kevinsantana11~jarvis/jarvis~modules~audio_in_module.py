import wave

import openai
import pyaudio

from .interfaces import InputModule


class AudioInputModule(InputModule):
    pyaudio_instance = pyaudio.PyAudio()
    CHANNELS = 1
    CHUNK = 1024
    RATE = 44100
    FORMAT = pyaudio.paInt16
    RECORD_SECONDS = 5

    def __call__(self) -> str:
        recording_file = wave.open("temp_recording.wav", "wb")
        self.pyaudio_instance.__init__()  # Initialize pyaudio incase it wasn't

        recording_file.setnchannels(self.CHANNELS)
        recording_file.setsampwidth(self.pyaudio_instance.get_sample_size(self.FORMAT))
        recording_file.setframerate(self.RATE)

        stream = self.pyaudio_instance.open(
            self.RATE, self.CHANNELS, self.FORMAT, input=True
        )
        print("[recording-start]")
        for _ in range(0, self.RATE // self.CHUNK * self.RECORD_SECONDS):
            recording_file.writeframes(stream.read(self.CHUNK))
        print("[recording-end]")

        stream.close()
        self.pyaudio_instance.terminate()  # Terminate the pyaudio instance

        recording_file.close()
        recording_file = open("temp_recording.wav", "rb")

        audio_transcribe = openai.Audio.transcribe("whisper-1", recording_file)
        transcribed_audio = audio_transcribe.get("text")
        recording_file.close()

        print("[transcription] - {}".format(transcribed_audio))
        return transcribed_audio
