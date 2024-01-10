import openai
from input.input_audio.transcriber.transcriber_interface import TranscriberInterface
from config import config

openai.api_key = config.OPENAI_API_KEY


class TranscriberOpenAi(TranscriberInterface):
    def transcribe(self, audio_filename) -> str:
        audio_file = open(audio_filename, "rb")
        transcript = openai.Audio.transcribe(
            "whisper-1",
            audio_file,
            language=config.LANGUAGE,
        )
        return transcript["text"]
