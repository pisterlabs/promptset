from openai import OpenAI


class WhisperAPI:
    def __init__(self, model) -> None:
        self.client = OpenAI()
        self.model = model
        self.response_format = "text"

    def transcribe(self, audio_file) -> str:
        af = open(audio_file, "rb")

        transcription = self.client.audio.transcriptions.create(
            model=self.model, file=af, response_format=self.response_format
        )

        af.close()

        return transcription
