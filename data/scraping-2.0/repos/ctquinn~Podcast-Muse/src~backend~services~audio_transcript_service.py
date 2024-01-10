from __future__ import annotations

import openai
import os
from dotenv import load_dotenv
from pathlib import Path

script_dir = Path(__file__).parent
project_root = script_dir.parent.parent.parent
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path)

openai_api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = openai_api_key

def create_transcript_file_stub(audio_path: str, transcript_output_path: str) -> str | None:
    """
    Creates a transcript file from the audio file at the input path and exports it to the output path
    :param audio_path: The path to the audio file to be transcribed
    :param transcript_output_path: The path to the transcript file to be created
    :return: The transcript text (if generated)
    """
    transcript_text = None
    print("Looking for audio at: " + audio_path)
    if not os.path.isfile(transcript_output_path):
        print("Audio Found, Transcribing...")
        trunc_audio_load = open(audio_path, "rb")

        audio_size = os.path.getsize(audio_path)
        file_size_mb = audio_size / (1024 ** 2)

        if file_size_mb < 25:
            transcript = openai.Audio.transcribe("whisper-1", trunc_audio_load)

            transcript_text = transcript.text
            print("File Transcribed:")
            print("#" * 50)
            print("Transcript: ", transcript_text)
            print("#" * 50)

            print("Now Saving Transcript")
            with open(transcript_output_path, 'w') as f:
                f.write(transcript_text)
    else:
        print("Transcript already exists")

    return transcript_text


def create_combined_transcript_file(audio_paths: list[str], transcript_output_path: str) -> str | None:
    """
    Creates a transcript file from the audio files at the input paths and exports it to the output path
    :param audio_paths: A list of paths to the audio files to be transcribed
    :param transcript_output_path: The path to the transcript file to be created
    :return: The transcript text (if generated)
    """
    # Checks if transcript already exists
    transcript_final_output_path = os.path.join(transcript_output_path, "transcript.txt")

    if os.path.isfile(transcript_final_output_path):
        print("Transcript already exists")
        return None

    transcript_text = ""
    for audio_path in audio_paths:
        print("Looking for audio at: " + audio_path)
        
        if not os.path.isfile(audio_path):
            print(f"No audio file found at {audio_path}. Skipping...")
            continue

        print("Audio Found, Transcribing...")
        trunc_audio_load = open(audio_path, "rb")

        audio_size = os.path.getsize(audio_path)
        file_size_mb = audio_size / (1024 ** 2)

        if file_size_mb < 25:
            transcript = openai.Audio.transcribe("whisper-1", trunc_audio_load)

            transcript_segment = transcript.text
            transcript_text += transcript_segment + " "

            print("File Transcribed:")
            print("#" * 50)
            print("Transcript: ", transcript_segment)
            print("#" * 50)
        else:
            print(f"The file size at {audio_path} is too large. Skipping...")

    print("Now Saving Transcript")
    # Create a new directory to store the output files
    # os.makedirs(transcript_output_path, exist_ok=True)

    # Set the transcript output path within the new directory

    with open(transcript_final_output_path, 'w') as f:
        f.write(transcript_text.strip())

    return transcript_text.strip() if transcript_text else None



# Loads transcript from file if not already developed above
def load_transcript_text(transcript_output_path: str) -> str | None:
    """
    Loads the transcript text from the transcript file at the input path
    :param transcript_output_path: The path to the transcript file to be loaded
    :return: The transcript text (if loaded)
    """
    transcript_text = None
    if not transcript_text:
        print("Now Loading Transcript")
        try:
            with open(transcript_output_path, 'r') as f:
                transcript_text = f.read()
                print("Transcript Loaded")
        except FileNotFoundError:
            print("Transcript file not found")
    else:
        print("Transcript already loaded")

    return transcript_text
