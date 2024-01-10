import os
import openai

# gets api key from env variable
openai.api_key = os.getenv("OPEN_AI_KEY")

class Converter:
    def __init__(self, file):
        self.file = file

    def captions(self):
        open_file = open(self.file, "rb")
        transcript = openai.Audio.transcribe("whisper-1",open_file )
        return transcript


    def big_file(self):
        file_size = os.path.getsize(self.file)
        # if(25000000 < file_size):




