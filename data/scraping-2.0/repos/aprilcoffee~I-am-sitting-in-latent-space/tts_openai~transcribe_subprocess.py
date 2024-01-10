import argparse
import openai
import os
import config 
# Define the OpenAI API key (you can pass it as a command-line argument)
# Replace with your actual OpenAI API key
openai_api_key = config.openai_api_key
# Set your OpenAI API key
openai.api_key = openai_api_key

def transcribe_audio(audio_file_path):
    try:
        with open(audio_file_path, 'rb') as audio_file:
            transcript = openai.Audio.transcribe("whisper-1", audio_file, language="de")
            print(transcript.text,end=' ')
    except Exception as e:
        print(f"Error during transcription: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Transcribe audio using OpenAI API")
    parser.add_argument("--audio_file", required=True, help="Path to the audio file for transcription")
    parser.add_argument("--api_key", required=True, help="OpenAI API key")

    args = parser.parse_args()

    # Set the OpenAI API key from the command-line argument
    openai.api_key = args.api_key

    # Transcribe the audio
    transcribe_audio(args.audio_file)

if __name__ == "__main__":
    main()
