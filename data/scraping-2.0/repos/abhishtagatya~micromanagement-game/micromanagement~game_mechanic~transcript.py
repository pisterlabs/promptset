from micromanagement.interface import OpenAIWhisper


class TranscriptMechanic:

    def __init__(self, token: str):
        self.openai_client = OpenAIWhisper(token=token)

    def transcribe(self, in_file, out_file):
        transcript = ""
        self.openai_client.convert_audio(in_file, out_file)
        with open(out_file, 'rb') as audio_file:
            transcript += self.openai_client.transcribe(audio_file).get('text', '')

        return transcript
