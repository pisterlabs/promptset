import os
import openai
from lib.Log import Log
from lib.default_config import default_config


Log.info("OpenAI Initializing")
openai.api_key = os.getenv("OPENAI_API_KEY")
Log.info("OpenAI Initialized")


class ApiBuilder:

    @staticmethod
    def ChatCompletion(msg: list[dict]):
        return openai.ChatCompletion.create(
            messages=msg,
            **vars(default_config.chatCompletionConfig)
        )

    @staticmethod
    def Image(prompt: str):
        Log.info("Picture Generating")
        res = openai.Image.create(
            prompt=prompt,
            **vars(default_config.imageConfig)
        )
        Log.info("Picture Generated")
        return res

    @staticmethod
    def Transcriptions(fileName: str):
        try:
            with open(fileName, "rb") as file:
                Log.info("Audio Translating")
                res = openai.Audio.translate(
                    file=file,
                    **vars(default_config.transcriptionsConfig)
                )
                Log.info("Audio Translated")
                return res
        except FileNotFoundError:
            Log.error("File Not Found")
