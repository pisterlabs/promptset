import os

import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def transcriptionsDemo():
    audio_file = open("zh.m4a", "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    print(transcript)


def translationsDemo():
    audio_file = open("zh.m4a", "rb")
    transcript = openai.Audio.translate("whisper-1", audio_file)
    print(transcript)


if __name__ == '__main__':
    # transcriptionsDemo()
    translationsDemo()
