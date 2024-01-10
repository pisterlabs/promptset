import openai

class SpeechRecognizer:
    def __init__(self, config):
        self.config = config
        openai.api_key = self.config.get_openai_api_key()

    def transcribe_speech(self, audio_file):
        response = openai.Speech.create(
            file=open(audio_file, "rb"),
            model="whisper",
            format="wav",
        )
        return response["choices"][0]["text"]
