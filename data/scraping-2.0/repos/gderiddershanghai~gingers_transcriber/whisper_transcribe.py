import os
import openai
from dotenv import load_dotenv, find_dotenv
from metrics import get_readability_features
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.getenv('OPENAI_API_KEY')


def transcribe(fp):
    try:
        # Using 'with' statement for automatic handling of file resources
        with open(fp, "rb") as audio_file:
            # Transcribe the audio using OpenAI's Whisper model in English
            # Assuming the response_format is 'text' for plain text transcription
            transcript = openai.Audio.transcribe("whisper-1", audio_file, language="en", response_format="text")
            additional_info = get_readability_features(transcript)
            print(transcript)

        # Save the transcript to a text file
        with open('transcript_whisper.txt', 'w', encoding='utf-8') as file:
            file.write(transcript+ additional_info)
            print("Transcript saved as 'transcript_whisper.txt'.")

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

    return 'transcript_whisper.txt'
