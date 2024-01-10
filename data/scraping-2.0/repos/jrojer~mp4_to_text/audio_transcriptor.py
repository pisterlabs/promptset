import env
import openai
from butter.checks import check_that, dir_for_file_exists, is_file

openai.api_key = env.OPENAI_API_KEY


class AudioTranscriptor:
    def __init__(self, audio_filepath: str, transcript_filepath: str):
        check_that(is_file(audio_filepath), f"File {audio_filepath} does not exist")
        check_that(
            dir_for_file_exists(transcript_filepath),
            f"Directory for {transcript_filepath} does not exist",
        )
        self.transcript_path = transcript_filepath
        self.audio_path = audio_filepath
        self.client = openai.OpenAI()

    def transcribe(self, prompt="") -> str:
        if is_file(self.transcript_path):
            print(f"File {self.transcript_path} already exists, skipping transcription")
            return

        with open(self.audio_path, "rb") as audio_file:
            print(f"Transcribing audio {self.audio_path}")
            transcription = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                prompt=prompt,
            )
        with open(self.transcript_path, "w") as transcript_file:
            transcript_file.write(transcription.model_dump_json())
        return transcription.text
