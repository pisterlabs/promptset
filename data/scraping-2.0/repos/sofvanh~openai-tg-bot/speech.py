import io
from openai import OpenAI


class SpeechGenerator:
    def __init__(self, openai_client: OpenAI):
        self.openai_client = openai_client

    def text_to_speech(self, text):
        # Split the text into chunks of 4096 characters each (OpenAI limit)
        # TODO This is probably going to cause some weirdness with the output since the AI will see the text start and end really abruptly
        chunks = [text[i:i+4096] for i in range(0, len(text), 4096)]
        fp = io.BytesIO()

        # TODO Can I add a progress indicator here somehow?
        for chunk in chunks:
            response = self.openai_client.audio.speech.create(
                model="tts-1",
                voice="onyx",
                input=chunk,
                response_format="opus"
            )
            for audio_chunk in response.iter_bytes():
                fp.write(audio_chunk)

        fp.seek(0)
        return fp
