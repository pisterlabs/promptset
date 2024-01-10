import openai

# v basic. transacribe audio file to text using the Whisper OpenAI API
class Transcriber:
    def __init__(self, api_key, model):
        openai.api_key = api_key
        self.model = model

    def transcribe_audio(self, audio_file_path):
        with open(audio_file_path, "rb") as audio_file:
            transcript = openai.Audio.transcribe(self.model, audio_file)
            return transcript["text"]
