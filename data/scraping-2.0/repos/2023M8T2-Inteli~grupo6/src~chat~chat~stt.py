import os
import rclpy
from rclpy.node import Node
import sounddevice as sd
import numpy as np
from openai import OpenAI
from pathlib import Path
import wave

class STT(Node):
    def __init__(self):
        super().__init__('audio_transcription_node')
        self.client = OpenAI()

    def record_audio(self, file_path, duration=10, sample_rate=44100):
        self.get_logger().info("Gravando...")

        audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=2, dtype=np.int16)
        sd.wait()

        with wave.open(file_path, 'wb') as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data.tobytes())

        self.get_logger().info(f"Audio recorded and saved to {file_path}")

    def transcribe_audio(self, file_path):
        with open(file_path, "rb") as file:
            transcription = self.client.audio.transcriptions.create(file=file)
        return transcription.text

    def start_transcription(self, file_path):
        if file_path:
            transcription = self.transcribe_audio(file_path)
        else:
            file_path = 'gravacao.wav'
            self.record_audio(file_path)
            transcription = self.transcribe_audio(file_path)

        self.get_logger().info('TRANSCRIPTION: %s' % transcription)

def main(args=None):
    rclpy.init(args=args)
    audio_transcription_node = STT()
    file_path = '/path/to/your/audio/file.wav'  # Set your default audio file path here

    try:
        audio_transcription_node.start_transcription(file_path)
    except KeyboardInterrupt:
        pass

    audio_transcription_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()