# Note: you need to be using OpenAI Python v0.27.0 for the code below to work

import openai
import os
import click
from dotenv import load_dotenv

load_dotenv()

# Set up OpenAI API credentials
openai.api_key = os.getenv("OPENAI_API_KEY")

@click.command()
@click.option(
    "--audio_file_name",
    default="deutsch",
    type=str,
    help="Name of the audio file in audio folder",
)
@click.option(
    "--methode",
    default="transcript",
    type=str,
    help="transcript or translate",
)
@click.option(
    "--target_language",
    default="en",
    type=str,
    help="target language for translation (Default: en)",
)
def main(audio_file_name: str, methode: str, target_language: str):
    audio_file = open(f"./audio/{audio_file_name}.mp3", "rb")

    if methode == "transcript":
        transcript_result = openai.Audio.transcribe("whisper-1", audio_file)
        print("transcript: ", transcript_result.text)
        
    elif methode == "translate":
        translate_result = openai.Audio.translate("whisper-1", audio_file, target_language)
        print("translate: ", translate_result.text)

if __name__ == "__main__":
    main()