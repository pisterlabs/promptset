import openai
import os
import sys


def main():
    openai.api_key = os.getenv("OPENAI_API_KEY")
    audio_file = open(sys.argv[1], "rb")

    response = openai.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file
    )

    print(response.text)


main()