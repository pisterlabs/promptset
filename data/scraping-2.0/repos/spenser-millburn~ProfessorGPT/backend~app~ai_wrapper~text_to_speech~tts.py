from pathlib import Path
import os
from openai import OpenAI
class TextToSpeech:
    def __init__(self):
        self.client = OpenAI()
        self.output_file_path = ""
        self.output_prefix = Path(os.getcwd()) / "output"

        if not self.output_prefix.exists():
            os.mkdir(self.output_prefix)


    def generate_speech(self, input_text, output_file_name , file_extension = "mp4"):
        self.output_file_path = self.output_prefix / (output_file_name + "." + file_extension)
        response = self.client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=input_text
        )
        resp = response.stream_to_file(self.output_file_path)
        return resp

    def get_output_file_path(self):
        return self.output_file_path
