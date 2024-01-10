import logging

import config
from openai import OpenAI

logger = logging.getLogger("ratemate")


class RateMate:
    def __init__(self):
        self.client = OpenAI(api_key=config.API_KEY)

    def rate_audio(self, audio_file_path: str) -> str:
        lyrics = self.transcribe_audio(audio_file_path)
        pegi_score = self.get_pegi_score(lyrics)

        return pegi_score

    def transcribe_audio(self, audio_file_path: str) -> str:
        logger.info(f"Processing audio file: {audio_file_path}")

        with open(audio_file_path, "rb") as audio_file:
            transcript = self.client.audio.transcriptions.create(
                model="whisper-1", file=audio_file
            )

            return transcript.text

    def get_pegi_score(self, lyrics: str) -> str:
        messages = [
            {
                "role": "system",
                "content": "You are a bot which takes a song lyrics as input and make a pegi score for it.",
            }
        ]
        logger.info(f"Song lyrics: {lyrics}")

        messages.append(
            {"role": "user", "content": lyrics},
        )
        chat = self.client.chat.completions.create(
            model="gpt-3.5-turbo", messages=messages
        )
        reply = chat.choices[0].message.content

        messages.append({"role": "assistant", "content": reply})

        return reply
