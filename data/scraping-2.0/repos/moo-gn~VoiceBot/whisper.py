import os
from io import BytesIO
from tempfile import NamedTemporaryFile
from credentials import OPENAI_API_TOKEN
import openai


openai.api_key = OPENAI_API_TOKEN


class Whisper:


    def __init__(self) -> None:
        self.model = "whisper-1"


    def _prepare_file_from_bytes(self, audio_file: BytesIO):
        temp = NamedTemporaryFile(suffix=".wav", delete=False)
        with open(temp.name, "wb") as f:
            f.write(audio_file.getbuffer())

        return temp


    def _clean_and_delete_file(self, audio_file) -> None:
        audio_file.close()
        os.unlink(audio_file.name)


    def get_transcription(self, audio_file: BytesIO):

        input_audio_file = self._prepare_file_from_bytes(audio_file)

        transcript = openai.Audio.transcribe(
            model=self.model, 
            file=open(input_audio_file.name, mode="rb"),
            response_format="verbose_json",
            prompt="this is an excerpt from a particular user in a voice chat, they may be talking in english or arabic, but try to parse all of the text to english",
            language="en"
        )

        return transcript
